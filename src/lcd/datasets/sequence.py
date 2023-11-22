import json
import random
from collections import namedtuple

import numpy as np
import torch

Batch = namedtuple("Batch", "trajectories conditions")
KwargsBatch = namedtuple("KwargsBatch", "batch kwargs")
ValueBatch = namedtuple("ValueBatch", "trajectories conditions values")


def cut_last(tensor):
    """
    Remove the last element along the first dimension of a tensor.

    Parameters:
    tensor (Tensor): A PyTorch tensor from which the first element will be removed.

    Returns:
    Tensor: A tensor with the same shape as the input, except with the first element
            along the first dimension removed.
    """
    return tensor[:-1, ...]


def cut_first(tensor):
    """
    Remove the first element along the first dimension of a tensor.

    Parameters:
    tensor (Tensor): A PyTorch tensor from which the first element will be removed.

    Returns:
    Tensor: A tensor with the same shape as the input, except with the first element
            along the first dimension removed.
    """
    return tensor[1:, ...]


def stack_next(tensor):
    """
    Stack the elements of a tensor along a new dimension after removing the first
    element from the original tensor and the last element from the original tensor.

    Parameters:
    tensor (Tensor): A PyTorch tensor which will be manipulated by removing the
                     first and last elements, then stacking them along a new dimension.

    Returns:
    Tensor: A tensor where the removed first elements and last elements of the original
            tensor are concatenated along a new dimension, effectively doubling
            the size of the second dimension while reducing the first dimension size.
    """
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


class ClevrDataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(
        self,
        *args,
        buf,
        lang_embeds,
        encoder_path,
        frame_offset=0,
        horizon=4,
        clip_stride=16,
        **kwargs,
    ):
        print(f"{buf=}")
        self.clip_stride = clip_stride
        self.frame_offset = frame_offset
        self.horizon = horizon
        self.buf = buf
        self.lang_embeds = torch.load(lang_embeds, map_location="cpu")
        # encoder = torch.load(encoder_path, map_location="cpu")
        # encoder_type = next(encoder.parameters())

        # def encode(buf):
        #     with torch.no_grad():
        #         buf["obs"] = encoder.encode(buf["obs"].to(encoder_type))
        #         buf["next_obs"] = encoder.encode(buf["next_obs"].to(encoder_type))

        # encode(self.buf)

        self.observation_dim = kwargs.get("observation_dim")
        self.action_dim = kwargs.get("action_dim")

    def __len__(self):
        return self.buf.shape[0]

    def __getitem__(self, index):
        lang = self.lang_embeds[str(self.buf["obs_goal"][index].int().tolist())]
        act = torch.nn.functional.one_hot(
            self.buf["actions"][index : index + 1], num_classes=40
        )
        obs = self.buf["obs"][index : index + 1].float()
        traj = torch.concat(
            (
                self.buf["next_obs"][index : index + 1],
                self.buf["obs"][index : index + 1],
            ),
            dim=1,
        )
        return KwargsBatch(
            Batch(traj, lang), {"inpaint": {0: traj[0, self.action_dim :]}}
        )



class ClevrEnd2EndDataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(
        self,
        *args,
        buf,
        lang_embeds,
        encoder_path,
        frame_offset=0,
        horizon=4,
        clip_stride=16,
        **kwargs,
    ):
        print(f"{buf=}")
        self.clip_stride = clip_stride
        self.frame_offset = frame_offset
        self.horizon = horizon
        self.buf = buf
        self.lang_embeds = torch.load(lang_embeds, map_location="cpu")
        # encoder = torch.load(encoder_path, map_location="cpu")
        # encoder_type = next(encoder.parameters())

        # def encode(buf):
        #     with torch.no_grad():
        #         buf["obs"] = encoder.encode(buf["obs"].to(encoder_type))
        #         buf["next_obs"] = encoder.encode(buf["next_obs"].to(encoder_type))

        # encode(self.buf)

        self.observation_dim = kwargs.get("observation_dim")
        self.action_dim = kwargs.get("action_dim")

    def __len__(self):
        return self.buf.shape[0]

    def __getitem__(self, index):
        lang = self.lang_embeds[str(self.buf["obs_goal"][index].int().tolist())]
        act = torch.nn.functional.one_hot(
            self.buf["actions"][index : index + 1], num_classes=40
        )
        obs = self.buf["obs"][index : index + 1].float()
        traj = torch.concat(
            (
                act.to(obs),
                obs,
            ),
            dim=1,
            
        )
        return KwargsBatch(
            Batch(traj, lang), {"inpaint": {0: traj[0, self.action_dim :]}}
        )
