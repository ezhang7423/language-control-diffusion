from pathlib import Path

import torch
from tensordict.tensordict import TensorDict as td

from lcd import DATA_PATH


def load_dataset(
    clevr: bool,
    exp_path: Path,
    val_ratio: float = 0.1,
    upscale: bool = False,
    upscaled_state_dim=None,
    shuffle=False,
    path=None,
    encoder=None,
):
    if path:
        data: dict = torch.load(path)
    else:
        if not clevr:
            # dict_keys(['actions', 'obs', 'rewards', 'dones', 'language_goal', 'obs_goal'])
            data: dict = torch.load(DATA_PATH / "kitchen_buf.pt", map_location="cpu")
        else:
            data: dict = torch.load(DATA_PATH / "ball_buf.pt", map_location="cpu")

    del data["language_goal"]
    
    data = td(data, data["actions"].shape[0])

    # shift obs
    data["next_obs"] = torch.cat((data["obs"][1:], data["obs"][-1:]), dim=0)


    if encoder:
        encoder_type = next(encoder.parameters())
        with torch.no_grad():
            data["obs"] = encoder.encode(data["obs"].to(encoder_type))
            data["next_obs"] = encoder.encode(data["next_obs"].to(encoder_type))

    if upscale:
        random = torch.randn((upscaled_state_dim, 10)).expand(
            data.shape[0], upscaled_state_dim, 10
        )
        torch.save(random, exp_path / "random_matrix_upscaler.pt")
        bmm = lambda obs: torch.bmm(random, obs.permute(0, 2, 1)).permute(0, 2, 1)
        data["obs"] = bmm(data["obs"])
        data["next_obs"] = bmm(data["next_obs"])

    # data['actions'] = torch.nn.functional.one_hot(data['actions'].squeeze().to(torch.int64), num_classes=72)
    data["actions"] = data["actions"].squeeze()

    ## shuffle
    total_len = data.shape[0]
    if shuffle:
        data = data[torch.randperm(total_len)]

    # split
    return data.split([int(total_len * (1 - val_ratio)), int(total_len * val_ratio)])
