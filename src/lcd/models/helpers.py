import math
import pdb

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

import lcd.utils as utils

# -----------------------------------------------------------------------------#
# ---------------------------------- modules ----------------------------------#
# -----------------------------------------------------------------------------#


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange("batch channels 1 horizon -> batch channels horizon"),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


# -----------------------------------------------------------------------------#
# --------------------------------- attention ---------------------------------#
# -----------------------------------------------------------------------------#


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class CrossAttention(nn.Module):  # replace with regular cross attention
    def __init__(
        self,
        query_dim,
        cross_attention_dim=None,
        heads=4,
        dim_head=32,
        bias=False,
        dropout=0,
    ):
        super().__init__()
        cross_attention_dim = (
            cross_attention_dim if cross_attention_dim is not None else query_dim
        )
        self.scale = dim_head**-0.5
        self.heads = heads
        self.query_dim = query_dim

        hidden_dim = dim_head * heads

        self.norm = LayerNorm(query_dim)
        self.to_q = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, hidden_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, hidden_dim, bias=bias)

        self.to_out = nn.Conv1d(hidden_dim, query_dim, 1)
        self.dropout = nn.Dropout(p=dropout) if dropout else nn.Identity()

    def forward(self, x, context=None):
        og_x = x
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: einops.rearrange(t.permute(0, 2, 1), "b (h c) d -> b h c d", h=self.heads), (q, k, v))  # type: ignore

        # b = batch
        # h = heads
        # c = channels
        # d = number of queries
        # e = number of key/values
        qk = (torch.einsum("b h c d, b h c e -> b h d e", q, k) * self.scale).softmax(
            dim=-1
        )

        out = torch.einsum("b h d e, b h c e -> b h c d", qk, v)
        out = einops.rearrange(out, "b h c d -> b (h c) d")
        return self.dropout(self.to_out(out) + og_x)


# -----------------------------------------------------------------------------#
# ---------------------------------- sampling ---------------------------------#
# -----------------------------------------------------------------------------#


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def apply_conditioning(x, conditions, action_dim):
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()
    return x


# -----------------------------------------------------------------------------#
# ---------------------------------- losses -----------------------------------#
# -----------------------------------------------------------------------------#


class WeightedLoss(nn.Module):
    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer("weights", weights)
        self.action_dim = action_dim

    def forward(self, pred, targ):
        """
        pred, targ : tensor
            [ batch_size x horizon x transition_dim ]
        """
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        a0_loss = (loss[:, 0, : self.action_dim] / self.weights[0, : self.action_dim]).mean()  # type: ignore
        return weighted_loss / self.weights[0, 0].detach(), {"a0_loss": a0_loss}

    def _loss(self, pred, targ):
        return NotImplementedError


class WeightedL1(WeightedLoss):
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


Losses = {"l1": WeightedL1, "l2": WeightedL2}
