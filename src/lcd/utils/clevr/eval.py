from typing_extensions import Annotated

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import torch
import tqdm
import typer
from eztils.torch import get_best_cuda
from rich import print
from torch import nn
from typer_config import conf_callback_factory, use_config
from typer_config.decorators import use_json_config
from typer_config.loaders import loader_transformer
from typer_config.utils import get_dict_section

from lcd import DATA_PATH
from lcd.models.mlp import ForwardModel
from lcd.utils.clevr import load_dataset
from lcd.utils.config import AttriDict

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

embeds = torch.load(DATA_PATH / "clevr_direct_embeddings.pt")


@dataclass
class EvalArgs:
    model: torch.nn.Module = None
    env: object = None
    only_llp: bool = False
    dataset: object = None
    num_sequences: int = 4
    max_episode_steps: int = 50
    silent: bool = False


def prepare_initial_observation(eval_arg: EvalArgs, ds_idx):
    if eval_arg.only_llp:
        sys.path.append("/home/ubuntu/talar/talar-openreview/talar")
        from serialize_scene_struct import deserialize

        struct = deserialize(eval_arg.dataset["scene_struct"][ds_idx].numpy())
        obs = eval_arg.env.envs[0].unwrapped.reset(scene_struct=struct)
    else:
        obs = eval_arg.env.envs[0].unwrapped.reset()
    return obs


def perform_evaluation_step(eval_arg: EvalArgs, obs, next_obs, ds_idx):
    action = eval_arg.model(obs, next_obs).argmax()[None]
    obs, rewards, dones, info = eval_arg.env.step(action)
    obs = obs[0]

    p_obs = preprocess_obs(obs)

    if eval_arg.only_llp:
        l1_error = (p_obs - next_obs).abs().mean()
        print("a_hat:", action)
        print("a:", eval_arg.dataset["actions"][ds_idx])
        print("l1 error:", (p_obs - next_obs).abs().mean())

    return dones


# 1. build mechanism design experimentation testbench


# default decorators
def use_json_config(
    section=None,
    param_name="config",
    param_help: str = "Configuration file.",
    default_value=None,
):
    callback = conf_callback_factory(
        loader_transformer(
            lambda f: json.loads(f),
            loader_conditional=lambda param_value: param_value,
            param_transformer=(
                lambda param_value: param_value if param_value else default_value
            )
            if default_value is not None
            else None,
            config_transformer=lambda config: get_dict_section(config, section),
        )
    )

    return use_config(callback=callback, param_name=param_name, param_help=param_help)


@app.command()
@use_json_config()
def eval_single_process(
    low_model_path: str = "/home/ubuntu/talar/lcd-iclr24-clevr/submodules/data/clevr/11-20_06:41:04.604072/model_30.pt",
    high_model_path: str = "/home/ubuntu/talar/lcd-iclr24-clevr/logs/diffusion/defaults_T20_S12/11-20_07:10:20/model_290000.pt",
    dataset_path: str = None,
    # env: object = None,
    num_sequences: int = 4,
    max_episode_steps: int = 50,
    silent: bool = False,
):
    only_llp = dataset_path is not None
    print(
        f"{low_model_path=}, {high_model_path=}, {dataset_path=}, {num_sequences=}, {max_episode_steps=}, {silent=}, {only_llp=}"
    )
    if only_llp:
        model = torch.load(low_model_path, map_location="cpu")
    else:
        model = DiffusionEvaluationWrapper(
            torch.load(high_model_path, map_location="cpu"),
            torch.load(low_model_path, map_location="cpu"),
        )

    device = get_best_cuda()
    model.to(device)

    eval_arg = EvalArgs(
        model=model,
        # env=env,
        only_llp=only_llp,
        dataset=load_dataset(
            clevr=True, upscale=False, exp_path=None, path=dataset_path
        )[-1]
        if dataset_path
        else None,
        num_sequences=num_sequences,
        max_episode_steps=max_episode_steps,
        silent=silent,
    )
    eval_arg.env = setup_env(eval_arg.max_episode_steps)

    success = []
    ds_idx = 0
    ds = eval_arg.dataset
    for i in range(eval_arg.num_sequences):
        obs = prepare_initial_observation(eval_arg, ds_idx)
        succ = False
        itr = range(eval_arg.max_episode_steps - 1)
        if not eval_arg.silent:
            itr = tqdm.tqdm(itr)
        for ts in itr:
            next_obs = (
                ds["next_obs"][ds_idx : ds_idx + 1] if eval_arg.only_llp else None
            )

            dones = perform_evaluation_step(eval_arg, obs, next_obs, ds_idx)
            if dones:
                succ = True
                break
            ds_idx += 1

        success.append(succ)
        print("Success:", succ, "Timesteps:", ts + 1)

    sr = sum(success) / len(success)
    print(sr, end="", file=sys.stderr)
    return sr


@dataclass
class DryEvalArgs:  # only basic types for calling in subprocess
    low_model_path: str = "/home/ubuntu/talar/lcd-iclr24-clevr/submodules/data/clevr/11-20_06:41:04.604072/model_30.pt"
    high_model_path: str = "/home/ubuntu/talar/lcd-iclr24-clevr/logs/diffusion/defaults_T20_S12/11-20_07:10:20/model_290000.pt"
    dataset_path: str = None
    # env: object = None
    num_sequences: int = 4
    max_episode_steps: int = 50
    silent: bool = True


def evaluate(
    dry_eval_args: DryEvalArgs,
    skip_eval: bool = False,
    num_processes: int = 25,
):
    if skip_eval:
        print("Skipping evaluation")
        return

    # if num_processes > 1:
    processes = []
    for _ in range(num_processes):
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--config",
                    json.dumps(vars(dry_eval_args)),  # Convert the args to JSON string
                ],
                stdout=subprocess.PIPE,  # Capture standard output
                stderr=subprocess.PIPE,  # Capture standard error
                # shell=True,
            )
        )
        time.sleep(0.2)

    print("Waiting for subprocesses to finish...")
    [p.wait() for p in processes]
    print("Processes finished")

    results = [float(p.communicate()[1].decode().split("\n")[-1]) for p in processes]
    return sum(results) / len(results), num_processes * dry_eval_args.num_sequences

    # print(result)
    # else:
    #     eval_single_process(**vars(dry_eval_args))


# def evaluate_llp(eval_arg: EvalArgs = None):
#     return evaluate_common(eval_arg, load_dataset=True)

# def evaluate(eval_arg: EvalArgs = None):
#     return evaluate_common(eval_arg)


def preprocess_obs(obs, return_goal=False):
    obs = torch.from_numpy(obs).float().unsqueeze(0)
    if return_goal:
        return obs[:, :-3], embeds[str(obs[0, -3:].int().tolist())]

    return obs[:, :-3]


class LLPEvaluationWrapper(nn.Module):
    def __init__(self, low) -> None:
        super().__init__()
        self.low: ForwardModel = low
        self.p = next(low.parameters())

    def forward(self, obs, next_obs):
        obs = preprocess_obs(obs)
        next_obs = preprocess_obs(next_obs)
        return self.low(obs.to(self.p), next_obs.to(self.p))


class DiffusionEvaluationWrapper(nn.Module):
    def __init__(self, high, low) -> None:
        super().__init__()
        self.high = high
        self.low: ForwardModel = low
        self.p = next(high.parameters())

    def forward(self, obs, _):
        obs, lang = preprocess_obs(obs, return_goal=True)

        # print(obs.shape, lang.shape)
        encoded_obs = self.low.encode(obs.to(self.p))
        samples = self.high.conditional_sample(
            lang.to(self.p)[None], horizon=1, inpaint={0: encoded_obs}
        ).trajectories
        goal = samples[:, :, :16].squeeze(dim=1)
        model_input = torch.concat((encoded_obs, goal), dim=-1).to(self.p)
        return self.low.final_layers(model_input)


def setup_env(max_episode_steps):
    # return Nop()

    sys.path.append("/home/ubuntu/talar/talar-openreview/talar")
    from envs.clevr_robot_env.env import LangGCPEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
    from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
    from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

    def make_env():
        def _thunk():
            env = LangGCPEnv(
                maximum_episode_steps=max_episode_steps,
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

    env = env_wrapper(1)
    env.reset()
    return env


if __name__ == "__main__":
    # typer.run(eval_single_process)
    app()

    # from eztils.torch import get_best_cuda, seed_everything

    # # seed_everything(0)

    # model = torch.load(
    #     "/home/ubuntu/talar/lcd-iclr24-clevr/submodules/data/clevr/11-20_06:41:04.604072/model_30.pt",
    #     map_location="cpu",
    # )
    # diffusion = torch.load(
    #     "/home/ubuntu/talar/lcd-iclr24-clevr/logs/diffusion/defaults_T20_S12/11-20_07:10:20/model_290000.pt",
    #     map_location="cpu",
    # )

    # dataset = load_dataset(
    #     clevr=True,
    #     upscale=False,
    #     exp_path=None,
    #     path="/home/ubuntu/talar/talar-openreview/buf_2023-11-20---02-05.pt",
    # )
    # max_episode_steps = 50

    # eval_arg = EvalArgs(
    #     model=DiffusionEvaluationWrapper(diffusion, model).to(device),
    #     skip_eval=False,
    #     env=setup_env(max_episode_steps),
    #     only_llp=False,
    #     num_sequences=4,
    #     num_processes=25,
    #     dataset=dataset[-1],  # use validation set
    #     max_episode_steps=max_episode_steps,
    #     silent=True,
    # )

    # sr = evaluate(eval_arg)

    # print(sr)
    # import wandb
    # config = vars(eval_arg)
    # config['model'] = str(config['model'])

    # wandb.init(
    #     project="test-sr",
    #     config=config,
    #     group='test-sr-100-2',
    # )
    # wandb.log({'sr': sr})
    # main(
    #     width=16384,
    #     depth=24,
    #     # batch_size=72,
    #     # skip_eval=True,
    #     # lr=2e-5,
    #     use_wandb=True
    # )