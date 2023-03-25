import copy
import os
import pdb

import einops
import numpy as np
import torch
import tqdm

import wandb
from lcd.datasets.sequence import Batch

from .arrays import batch_to_device
from .timer import Timer


def cycle(dl):
    while True:
        yield from dl


def wlog(*args, **kwargs):
    if wandb.run is not None:
        wandb.log(*args, **kwargs)


class EMA:
    """
    empirical moving average
    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer:
    def __init__(
        self,
        diffusion_model,
        dataset,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder="./results",
        n_reference=8,
        bucket=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(
            torch.utils.data.DataLoader(
                self.dataset,
                batch_size=train_batch_size,
                num_workers=1,
                shuffle=True,
                pin_memory=True,
            )
        )
        self.dataloader_vis = cycle(
            torch.utils.data.DataLoader(
                self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
            )
        )
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    # -----------------------------------------------------------------------------#
    # ------------------------------------ api ------------------------------------#
    # -----------------------------------------------------------------------------#

    def train(self, n_train_steps):
        timer = Timer()
        for step in tqdm.tqdm(range(n_train_steps)):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch)

                if isinstance(batch, Batch):
                    loss, infos = self.model.loss(*batch)
                else:
                    loss, infos = self.model.loss(*batch[0], **batch[1])
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            wlog({"loss": loss, **infos})

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                self.save(self.step)

            if self.step % self.log_freq == 0:
                infos_str = " | ".join(
                    [f"{key}: {val:8.4f}" for key, val in infos.items()]
                )
                print(
                    f"{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}",
                    flush=True,
                )

            if self.sample_freq and self.step % self.sample_freq == 0:
                samples = self.ema_model.conditional_sample(
                    batch[0][1], horizon=1, **batch[1]
                ).trajectories
                l1 = (samples[:, :, :32] - batch[0][0][:, :, :32]).abs()
                wlog({"generation_mae_mean": l1.mean(), "generation_mae_std": l1.std()})

            self.step += 1

    def save(self, epoch):
        """
        saves model and ema to disk;
        syncs to storage bucket if a bucket is specified
        """
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
        }
        savepath = os.path.join(self.logdir, f"state_{epoch}.pt")
        torch.save(data, savepath)
        torch.save(self.ema_model, os.path.join(self.logdir, f"model_{epoch}.pt"))
        print(f"[ utils/training ] Saved model to {savepath}", flush=True)

    def load(self, epoch):
        """
        loads model and ema from disk
        """
        loadpath = os.path.join(self.logdir, f"state_{epoch}.pt")
        data = torch.load(loadpath)

        self.step = data["step"]
        self.model.load_state_dict(data["model"])
        self.ema_model.load_state_dict(data["ema"])
