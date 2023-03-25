import json
import random
from collections import namedtuple

import numpy as np
import torch

Batch = namedtuple("Batch", "trajectories conditions")
KwargsBatch = namedtuple("KwargsBatch", "batch kwargs")
ValueBatch = namedtuple("ValueBatch", "trajectories conditions values")


def cut_last(tensor):
    return tensor[:-1, ...]


def cut_first(tensor):
    return tensor[1:, ...]


def stack_next(tensor):
    return torch.concat((cut_first(tensor), cut_last(tensor)), dim=1)


class HulcDataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(
        self,
        *args,
        buf,
        task_to_ann,
        lang_embeds,
        frame_offset=0,
        horizon=4,
        clip_stride=16,
        **kwargs,
    ):
        print(f"{buf=}")
        self.clip_stride = clip_stride
        self.frame_offset = frame_offset
        self.horizon = horizon

        print("loading dataset...")
        buf = torch.load(buf, map_location="cpu")
        print("done loading!")

        buf["states"] = np.array([i[-100:] for i in buf["states"]])
        lens = np.array([len(f) for f in buf["states"]])
        valid_indices = np.argwhere(lens >= 17).squeeze()

        self.buf = {
            "states": buf["states"][valid_indices],
            "goal_lang": np.array(buf["goal_lang"])[valid_indices],
            "goal_task": np.array(buf["goal_task"])[valid_indices],
        }
        self.task_to_ann = json.load(open(task_to_ann))
        self.lang_embeds = torch.load(lang_embeds)

        self.observation_dim = kwargs.get("observation_dim")
        self.action_dim = kwargs.get("action_dim")

    def __len__(self):
        return len(self.buf["states"])

    def __getitem__(self, index):
        obs_traj = self.buf["states"][index]
        idx = random.randint(0, len(obs_traj) - self.clip_stride - 1)
        offset = random.randint(-self.frame_offset, self.frame_offset)

        indices = (
            idx + torch.arange(self.horizon + 1) * self.clip_stride + offset
        ).clamp(max=len(obs_traj) - 1)

        traj = stack_next(obs_traj[indices])
        lang = self.lang_embeds[
            random.choice(self.task_to_ann[self.buf["goal_task"][index]])
        ]
        return KwargsBatch(
            Batch(traj, lang), {"inpaint": {0: traj[0, self.action_dim :]}}
        )
