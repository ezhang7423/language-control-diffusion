# %%
import datetime
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import tqdm
from eztils.torch.wandb import log_wandb_distribution
from tensordict.tensordict import TensorDict as td
from torch import nn
from torchinfo import summary

import wandb
from lcd import DATA_PATH
from lcd.models.mlp import ForwardModel
from lcd.utils.config import AttriDict
from lcd.utils.eval import evaluate_policy, print_and_save
from lcd.utils.setup import set_seed
from lcd.utils.training import cycle
from rich import print

def get_best_cuda() -> int:
    import numpy as np
    import pynvml

    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    deviceMemory = []
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        deviceMemory.append(mem_info.free)
    deviceMemory = np.array(deviceMemory, dtype=np.int64)
    best_device_index = np.argmax(deviceMemory)
    print("best gpu:", best_device_index)
    return str(best_device_index.item())


def wlog(*args, **kwargs):
    if wandb.run is not None:
        wandb.log(*args, **kwargs)
    else:
        print(*args, **kwargs)


@dataclass
class EvalArgs:
    args: AttriDict = None
    model: torch.nn.Module = None
    skip_eval: bool = False
    env: object = None
    use_dataset: bool = False
    dataset: object = None
    num_sequences: int = 1000


def main(
    seed: int = 12,
    width: int = 256,
    upscaled_state_dim: int = 2048,
    upscale: bool = False,
    depth: int = 2,
    num_epochs: int = 100,
    num_steps: int = int(1e4),
    batch_size: int = 512,
    lr: float = 2e-4,
    diffusion_dim: int = 16,
    use_wandb: bool = False,
    skip_eval: bool = True,
    clevr: bool = True,
):
    set_seed(seed)

    args, _, _, values = inspect.getargvalues(inspect.currentframe())
    args = AttriDict({k: v for k, v in values.items() if k in args})

    # setup
    env_name = "clevr" if clevr else "kitchen"
    args.env_name = env_name

    exp_path = (
        DATA_PATH / env_name / datetime.datetime.now().strftime("%m-%d_%H:%M:%S.%f")
    )
    args.log_dir = exp_path
    exp_path.mkdir(parents=True)
    print("Saving to", exp_path)
    if use_wandb:
        wandb.init(
            project="vanilla-diffuser",
            entity="lang-diffusion",
            name=f"{env_name}-ce",
            config=vars(args),
        )

    if not upscale:
        upscaled_state_dim = 10
    # model
    final_dim = 40 if clevr else None
    model = ForwardModel(
        in_dim=upscaled_state_dim * 2, final_dim=final_dim, diffusion_dim=diffusion_dim, num_units=[width] * depth
    )
    summary(model)

    # dataset
    train_dataset, val_dataset = load_dataset(
        clevr=clevr,
        exp_path=exp_path,
        upscaled_state_dim=upscaled_state_dim,
        upscale=upscale,
        shuffle=True,
    )

    def get_dataloader(dataset):
        return cycle(
            torch.utils.data.DataLoader(
                dataset,
                batch_size=int(batch_size),  # avg non pad length around 10?
                num_workers=1,
                shuffle=True,
                # pin_memory=True,
                collate_fn=lambda x: x,
            )
        )

    train_dataloader = get_dataloader(train_dataset)
    val_dataloader = get_dataloader(val_dataset)
    eval_arg = EvalArgs(args=args, model=model, skip_eval=skip_eval, env=setup_env() if not skip_eval else None)

    # test evaluation is working
    evaluate(eval_arg)

    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
    device = "cuda:" + get_best_cuda()
    model.to(device)
    # %%
    # if True:

    model.train()
    from torch.nn import functional as F

    for epoch in range(num_epochs):
        # for epoch in range(1):
        print("*" * 50)
        print(f"{epoch=}")
        print("*" * 50)
        torch.save(model, exp_path / f"model_{epoch}.pt")
        torch.save(optimizer, exp_path / f"optimizer_{epoch}.pt")
        for i in tqdm.tqdm(range(num_steps)):
            # for i in range(100):
            def get_loss(dataloader, log_val=False):
                # batch = preprocess_batch(next(dataloader), device)
                extra_stats = {}
                batch = next(dataloader).to(device).float()
                pred = model.forward(
                    torch.concat((batch["obs"], batch["next_obs"]), dim=-1)
                ).to(device)

                act_int = batch["actions"].to(torch.int64)
                loss = F.cross_entropy(pred, act_int)
                if log_val:
                    correct = pred.argmax(dim=1) == act_int
                    extra_stats['eval/acc'] = sum(correct) / len(correct)
                    log_wandb_distribution("predictions", pred.reshape(-1))
                # print(loss)
                return loss, extra_stats

            loss, stats = get_loss(train_dataloader)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if use_wandb:
                wlog({"train/loss": loss, **stats})
            if not (i % 100):
                val_loss, val_stats = get_loss(val_dataloader, log_val=True)
                wlog({"eval/loss": val_loss, **val_stats})

        if not (epoch % 5):
            eval_arg.num_sequences = 10
            evaluate(eval_arg)

    eval_arg.num_sequences = 1000
    evaluate(eval_arg)


# %%


def load_dataset(
    clevr: bool,
    exp_path: Path,
    val_ratio: float = 0.1,
    upscale: bool = False,
    upscaled_state_dim=None,
    shuffle=False,
    path=None,
):  
    if path:
        data: dict = torch.load(path)
    else:
        if not clevr:
            # dict_keys(['actions', 'obs', 'rewards', 'dones', 'language_goal', 'obs_goal'])
            data: dict = torch.load(DATA_PATH / "kitchen_buf.pt")
        else:
            data: dict = torch.load(DATA_PATH / "ball_buf.pt")

    del data["language_goal"]
    data = td(data, data["actions"].shape[0])

    # shift obs
    data["next_obs"] = torch.cat((data["obs"][1:], data["obs"][-1:]), dim=0)

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


# %%


def setup_env():
    # return Nop()

    sys.path.append(
        "/home/ubuntu/talar/talar-openreview/talar"
    )
    from envs.clevr_robot_env.env import LangGCPEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
    from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
    from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
    
    def make_env():
        def _thunk():
            env = LangGCPEnv(
                maximum_episode_steps=50,
                action_type="perfect",
                obs_type="order_invariant",
                direct_obs=True,
                use_subset_instruction=True,
                fail_dist=0.2,
                language_model_type="policy_ag",
                # use_camera=True , # TODO debugging
                
                
                mode="train",  # TODO put in test mode instead of train
            )

            env = Monitor(env, None, allow_early_resets=True)

            return env

        return _thunk

    def env_wrapper(num):
        envs = [make_env() for _ in range(num)]

        if len(envs) > 1:
            envs = SubprocVecEnv(envs)
        else:
            envs = DummyVecEnv(envs)

        envs = VecNormalize(envs, norm_reward=True, norm_obs=False, training=False)

        return envs

    env =  env_wrapper(1)
    env.reset()
    return env

def preprocess_obs(obs):
    return torch.from_numpy(obs[:-3]).float().unsqueeze(0)

def evaluate(eval_arg: EvalArgs = None):
    if eval_arg.skip_eval:
        print("Skipping evaluation")
        return

    from serialize_scene_struct import deserialize
    success = []
    rewards = []
    
    ds_idx = 0
    ds = eval_arg.dataset
    
    p = next(eval_arg.model.parameters())
    
    for i in range(eval_arg.num_sequences):
        struct = deserialize(ds['scene_struct'][ds_idx].numpy())
        # obs = preprocess_obs(eval_arg.env.reset(scene_struct=struct))
        obs = preprocess_obs(eval_arg.env.envs[0].unwrapped.reset(scene_struct=struct))
        
        for ts in range(10):
            next_obs = ds['next_obs'][ds_idx:ds_idx+1]
            model_input = torch.concat((obs, next_obs), dim=-1).to(p)
            action = eval_arg.model(model_input).argmax()[None]
            obs, rewards, dones, info = eval_arg.env.step(action)
            obs = preprocess_obs(obs[0])
            print('a_hat:', action)
            print('a:', ds['actions'][ds_idx])
            print('l1 error:', (obs - next_obs).abs().mean())
            ds_idx += 1
            if dones:
                success.append(True)
                break
            success.append(False)
    sr = sum(success) / len(success)
    print("Success rate: ", sr)
    return sr


class Nop(object):
    def nop(*args, **kw):
        pass

    def __getattr__(self, _):
        return self.nop


if __name__ == "__main__":
    model = torch.load(
        "/home/ubuntu/talar/lcd-iclr24-clevr/submodules/data/clevr/11-19_18:42:51.252737/model_99.pt",
        map_location="cpu",
    )
    dataset = load_dataset(clevr=True, upscale=False, exp_path=None, path = '/home/ubuntu/talar/talar-openreview/buf_2023-11-20---02-05.pt')

    eval_arg = EvalArgs(
        args={},
        model=model,
        skip_eval=False,
        env=setup_env(),
        use_dataset=True,
        dataset=dataset[-1], # use validation set
    )

    evaluate(eval_arg)
    # main(
    #     width=16384,
    #     depth=24,
    #     # batch_size=72,
    #     # skip_eval=True,
    #     # lr=2e-5,
    #     use_wandb=True
    # )
