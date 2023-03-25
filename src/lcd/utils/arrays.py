import collections
import pdb

import numpy as np
import torch

DTYPE = torch.float
DEVICE = "cuda:0"

# -----------------------------------------------------------------------------#
# ------------------------------ numpy <--> torch -----------------------------#
# -----------------------------------------------------------------------------#


def to_np(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x


def to_torch(x, dtype=None, device=None):
    dtype = dtype or DTYPE
    device = device or DEVICE
    if type(x) is dict:
        return {k: to_torch(v, dtype, device) for k, v in x.items()}
    elif torch.is_tensor(x):
        return x.to(device).type(dtype)
    return torch.tensor(x, dtype=dtype, device=device)


def to_device(x, device=DEVICE):
    if torch.is_tensor(x):
        return x.to(device)
    elif type(x) is dict:
        return {k: to_device(v, device) for k, v in x.items()}
    elif isinstance(x, tuple):
        return tuple(to_device(v, device) for v in x)
    elif isinstance(x, list):
        return [to_device(v, device) for v in x]
    else:
        raise RuntimeError(f"Unrecognized type in `to_device`: {type(x)}")


def apply_dict(fn, d, *args, **kwargs):
    return {k: fn(v, *args, **kwargs) for k, v in d.items()}


def set_device(device):
    if "cuda" in device:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)  # type: ignore


def batch_to_device(batch, device="cuda:0"):
    vals = [to_device(getattr(batch, field), device) for field in batch._fields]
    return type(batch)(*vals)


def _to_str(num):
    if num >= 1e6:
        return f"{(num/1e6):.2f} M"
    else:
        return f"{(num/1e3):.2f} k"


# -----------------------------------------------------------------------------#
# ----------------------------- parameter counting ----------------------------#
# -----------------------------------------------------------------------------#


def param_to_module(param):
    module_name = param[::-1].split(".", maxsplit=1)[-1][::-1]
    return module_name


def report_parameters(model, topk=10):
    counts = {k: p.numel() for k, p in model.named_parameters()}
    n_parameters = sum(counts.values())
    print(f"[ utils/arrays ] Total parameters: {_to_str(n_parameters)}")

    modules = dict(model.named_modules())
    sorted_keys = sorted(counts, key=lambda x: -counts[x])  # type: ignore
    max_length = max([len(k) for k in sorted_keys])
    for i in range(topk):
        key = sorted_keys[i]
        count = counts[key]
        module = param_to_module(key)
        print(" " * 8, f"{key:10}: {_to_str(count)} | {modules[module]}")

    remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])
    print(
        " " * 8,
        f"... and {len(counts)-topk} others accounting for {_to_str(remaining_parameters)} parameters",
    )
    return n_parameters
