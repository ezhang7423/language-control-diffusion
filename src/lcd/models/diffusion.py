import pdb
from collections import namedtuple

import numpy as np
import torch
from torch import nn

import lcd.utils as utils

from .helpers import Losses, apply_conditioning, cosine_beta_schedule, extract

Sample = namedtuple("Sample", "trajectories values chains")


@torch.no_grad()
def default_sample_fn(model, x, cond, t, **kwargs):
    model_mean, _, model_log_variance = model.p_mean_variance(
        x=x, cond=cond, t=t, **kwargs
    )
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        horizon,
        observation_dim,
        action_dim,
        n_timesteps=1000,
        loss_type="l1",
        clip_denoised=False,
        predict_epsilon=True,
        action_weight=1.0,
        loss_discount=1.0,
        loss_weights=None,
        classifier_drop_probability=-1.0,  # 0.1
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        self.classifier_drop_probability = classifier_drop_probability
        betas = cosine_beta_schedule(n_timesteps)
        alphas: torch.Tensor = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        """
        sets loss coefficients for trajectory

        action_weight   : float
            coefficient on first action loss
        discount   : float
            multiplies t^th timestep of trajectory loss by discount**t
        weights_dict    : dict
            { i: c } multiplies dimension i of observation loss by c
        """
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None:
            weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum("h,t->ht", discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, : self.action_dim] = action_weight
        return loss_weights

    # ------------------------------------------ sampling ------------------------------------------#
    def predict_noise_from_start(self, x_t, t, x0):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return x0
        else:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
            ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_start_from_noise(self, x_t, t, noise):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        cond,
        prior=None,
        inpaint=None,
        verbose=False,
        return_chain=False,
        sample_fn=default_sample_fn,
        **sample_kwargs,
    ):
        device = self.betas.device

        batch_size = shape[0]
        if prior is None:
            x = torch.randn(shape, device=device)  # type: ignore
        else:
            x = prior

        if inpaint is not None:
            x = apply_conditioning(x, inpaint, self.action_dim)

        chain = [x]

        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, cond, t, **sample_kwargs)
            if inpaint is not None:
                x = apply_conditioning(x, inpaint, self.action_dim)

            if return_chain:
                chain.append(x)

        # x, values = sort_by_values(x, values)
        if return_chain:
            chain = torch.stack(chain, dim=1)  # type: ignore
        return Sample(x, values, chain)

    @torch.no_grad()
    def ddim_sample(
        self, shape, cond, prior=None, inpaint=None, return_chain=False, **sample_kwargs
    ):
        batch_size, device, total_timesteps, sampling_timesteps, eta = (
            shape[0],
            self.betas.device,
            self.n_timesteps,
            10,
            0,
        )

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device=device)  # type: ignore
        if prior is not None:
            x += 0.02 * prior

        if inpaint is not None:
            x = apply_conditioning(x, inpaint, self.action_dim)

        x_start = None

        chain = []

        for time, time_next in time_pairs:
            t = make_timesteps(batch_size, time, device)
            t_next = make_timesteps(batch_size, time_next, device)

            model_out = self.model(x, cond, t)
            x_start = self.predict_start_from_noise(x, t=t, noise=model_out)
            pred_noise = self.predict_noise_from_start(x, t=t, x0=model_out)

            if time_next < 0:
                x = x_start
                continue

            alpha = extract(self.alphas_cumprod, t, x.shape)
            alpha_next = extract(self.alphas_cumprod, t_next, x.shape)

            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            if inpaint is not None:
                x = apply_conditioning(x, inpaint, self.action_dim)

            if return_chain:
                chain.append(x)

        if return_chain:
            chain = torch.stack(chain, dim=1)  # type: ignore

        if inpaint is not None:
            x = apply_conditioning(x, inpaint, self.action_dim)

        return Sample(x, None, chain)

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, **sample_kwargs):
        """
        conditions : [ (time, state), ... ]
        """
        batch_size = len(cond)
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        if sample_kwargs.get("ddim"):
            return self.ddim_sample(shape, cond, **sample_kwargs)

        return self.p_sample_loop(shape, cond, **sample_kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, inpaint=None, **model_kwargs):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise).to(x_start)
        if inpaint is not None:
            x_noisy = apply_conditioning(x_noisy, inpaint, self.action_dim)

        if np.random.uniform() < self.classifier_drop_probability:
            cond = torch.zeros_like(cond)

        x_recon = self.model(x_noisy, cond, t, **model_kwargs)
        if inpaint is not None:
            x_recon = apply_conditioning(x_recon, inpaint, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond, inpaint=None, **model_kwargs):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t, inpaint=inpaint, **model_kwargs)

    def forward(self, *args, **kwargs):
        return self.conditional_sample(*args, **kwargs)
