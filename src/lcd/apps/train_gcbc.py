# %%

print('runnning')

import datetime
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import tqdm
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
from eztils.torch.wandb import log_wandb_distribution

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


device = "cuda:" + get_best_cuda()


def wlog(*args, **kwargs):
    if wandb.run is not None:
        wandb.log(*args, **kwargs)
    else:
        print(*args, **kwargs)


@dataclass
class EvalArgs:
    args: AttriDict = None
    eval_state: AttriDict = None
    model: torch.nn.Module = None
    skip_eval: bool = False
    num_sequences: int = 1000


def main(
    seed: int = 12,
    width: int = 64,
    upscaled_state_dim: int = 2048,
    upscale: bool = False,
    depth: int = 3,
    num_epochs: int = 100,
    num_steps: int = int(1e4),
    batch_size: int = 512,
    lr: float = 2e-4,
    use_wandb: bool = False,
    skip_eval: bool = False,
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
    model = ForwardModel(in_dim=upscaled_state_dim * 2, final_dim=final_dim, num_units=[width] * depth)
    # model = torch.load(
    #     "/home/ubuntu/talar/lcd-iclr24-clevr/submodules/data/clevr/11-17_22:33:28.679430/model_17.pt",
    #     map_location="cpu",
    # )
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

    # speed test
    # for i in tqdm.tqdm(range(10000)): # around 120 it/s
    #     batch = next(train_dataloader)

    eval_arg = EvalArgs(args=args, eval_state=None, model=model, skip_eval=skip_eval)
    eval_callback = setup_evaluation(eval_arg)

    # validation evaluation is working
    eval_callback.on_training_start(locals(), globals())  #!!

    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
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
            def get_loss(dataloader, log_preds=False):
                # batch = preprocess_batch(next(dataloader), device)
                batch = next(dataloader).to(device).float()
                pred = model.forward(
                    torch.concat(
                        (batch["obs"], batch["next_obs"]), dim=-1
                    )
                ).to(device)
                # loss = ((pred - batch['actions']) ** 2).mean()
                loss = F.cross_entropy(pred, batch["actions"].to(torch.int64))
                if log_preds:
                    log_wandb_distribution('predictions', pred.reshape(-1))
                # print(loss)
                return loss, {}

            loss, stats = get_loss(train_dataloader)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if use_wandb:
                wlog({"train/loss": loss, **stats})
            if not (i % 100):
                val_loss, val_stats = get_loss(val_dataloader, log_preds=True)
                wlog({"eval/loss": val_loss, **val_stats})

        if not (epoch % 5):
            eval_arg.num_sequences = 10
            eval_callback.on_step()

    eval_arg.num_sequences = 1000
    eval_callback.on_step()


# %%



def load_dataset(
    clevr: bool,
    upscaled_state_dim: int,
    exp_path: Path,
    val_ratio: float = 0.1,
    upscale: bool = False,
    shuffle=False,
):
    
    if not clevr:
        # dict_keys(['actions', 'obs', 'rewards', 'dones', 'language_goal', 'obs_goal'])
        data: dict = torch.load(DATA_PATH / "kitchen_buf.pt")
    else:
        data: dict = torch.load(DATA_PATH / "ball_buf.pt")
        
    del data['language_goal']                
    data = td(data, data['actions'].shape[0])

    # shift obs
    data['next_obs'] = torch.cat((data['obs'][1:], data['obs'][-1:]), dim=0)
    
    # data["goals"] = data["goals"][:, None].repeat(1, 50, 1)
    # data.batch_size = data["goals"].shape[:2]
    random = torch.randn((upscaled_state_dim, 10)).expand(
        data.shape[0], upscaled_state_dim, 10
    )
    torch.save(random, exp_path / "random_matrix_upscaler.pt")

    if upscale:
        bmm = lambda obs: torch.bmm(random, obs.permute(0, 2, 1)).permute(0, 2, 1)
        data["obs"] = bmm(data["obs"])
        data["next_obs"] = bmm(data["next_obs"])

    # data['actions'] = torch.nn.functional.one_hot(data['actions'].squeeze().to(torch.int64), num_classes=72)
    data["actions"] = data["actions"].squeeze()

    #! add transform on states to upscale to 4096 or something

    ## shuffle
    total_len = data.shape[0]
    if shuffle:
        data = data[torch.randperm(total_len)]

    # split
    return data.split([int(total_len * (1 - val_ratio)), int(total_len * val_ratio)])


# %%

# if __name__ == "__main__":

#     main(
#         width=16384,
#         depth=24,
#         # batch_size=72,
#         # skip_eval=True,
#         # lr=2e-5,
#         use_wandb=True
#     )


class Nop(object):
    def nop(*args, **kw):
        pass

    def __getattr__(self, _):
        return self.nop


def setup_evaluation(eval_arg: EvalArgs = None):
    return Nop()

    sys.path.append(
        "/home/ubuntu/talar/lcd-iclr24-clevr/submodules/talar-openreview-fork/talar"
    )
    from envs.clevr_robot_env.env import LangGCPEnv
    from stable_baselines3.common.callbacks import CallbackList, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
    from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
    from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

    def make_env(mode):
        def _thunk():
            env = LangGCPEnv(
                maximum_episode_steps=50,
                action_type="perfect",
                obs_type="order_invariant",
                direct_obs=True,
                use_subset_instruction=True,
                fail_dist=0.20,
                language_model_type="bert_cont",
                mode=mode,
            )

            env = Monitor(env, None, allow_early_resets=True)

            return env

        return _thunk

    def env_wrapper(mode, num):  #! extend to franka
        envs = [make_env(mode) for _ in range(num)]

        if len(envs) > 1:
            envs = SubprocVecEnv(envs)
        else:
            envs = DummyVecEnv(envs)

        envs = VecNormalize(envs, norm_reward=True, norm_obs=False, training=False)

        return envs

        train_env = env_wrapper("train", 2)

    class EvaluationWrapper:
        def __init__(self, model, env) -> None:
            self.model = model
            self.env = env
            from stable_baselines3.common.vec_env import unwrap_vec_normalize

            self._vec_normalize_env = unwrap_vec_normalize(env)

        def get_env(self):
            return self.env

        def logger(self, *args, **kwargs):
            pass

        def predict(self, obs, state=None, mask=None, deterministic=False):
            obs = torch.tensor(obs).to(device)
            obs = obs.view(1, -1)
            pred = self.model.forward(  #! fix this
                obs[:, 32:], obs[:, :32].repeat_interleave(4, dim=0)
            ).to(device)
            return pred.cpu().detach().numpy()

        def get_vec_normalize_env(self):
            return self._vec_normalize_env

        def save(self, path):
            torch.save(self.model, path)

    test_env = env_wrapper("test", 2)
    error_env = env_wrapper("error", 2)

    train_callback = EvalCallback(  #! do we really need three?
        eval_env=train_env,
        log_path=eval_arg.args.log_dir / "train_eval",
        deterministic=False,
        eval_freq=1,
        n_eval_episodes=eval_arg.num_sequences,
        name="train",
    )
    test_callback = EvalCallback(
        eval_env=test_env,
        log_path=eval_arg.args.log_dir / "test_eval",
        deterministic=False,
        eval_freq=1,
        n_eval_episodes=eval_arg.num_sequences,
        name="test",
    )
    error_callback = EvalCallback(
        eval_env=error_env,
        log_path=eval_arg.args.log_dir / "error_eval",
        deterministic=False,
        eval_freq=1,
        n_eval_episodes=eval_arg.num_sequences,
        name="error",
    )

    cb_list = [
        train_callback,
        test_callback,
        error_callback,
    ]

    for cb in cb_list:
        cb.init_callback(EvaluationWrapper(eval_arg.model, cb.eval_env))

    return CallbackList(cb_list)
    # run evaluation
    eval_arg.model.eval()
    if eval_arg.skip_eval:
        return

    args = eval_arg.args.copy()
    args.update(
        {
            "num_sequences": eval_arg.num_sequences,
            "dm": TransformerEvaluationWrapper(eval_arg.model, device=device),
            "ep_len": 360,
            "subgoal_interval": 16,
        }
    )

    results, histories = evaluate_policy(eval_arg.eval_state, args)
    model_id = f"transformer-ablation-{args}"

    eval_results = print_and_save(results, args, histories, model_id)

    wlog({f"eval/{k}": v for k, v in eval_results.items()})
    eval_arg.model.train()


# %%


def preprocess_batch(batch, device):
    batch = batch.flatten(end_dim=1).to(device)

    # Find non-zero observation and next_observation entries
    non_zero_obs = torch.any(batch["obs"] != 0, dim=-1)
    non_zero_next_obs = torch.any(batch["next_obs"] != 0, dim=-1)

    non_padded_flat_mask = non_zero_obs.flatten() & non_zero_next_obs.flatten()

    return batch[non_padded_flat_mask]
    # Provided the model operates on a single time step of obs,
    # you can now index directly without reshaping:
    non_padded_obs = batch["obs"][non_padded_flat_mask, :]
    non_padded_next_obs = batch["next_obs"][non_padded_flat_mask, :]

    # Combine both masks to find where both conditions are true
    non_padded_mask = non_zero_obs & non_zero_next_obs

    # Use the mask to filter out the padded values in each tensor field
    filtered_batch = {
        key: torch.masked_select(value, non_padded_mask.unsqueeze(-1))
        for key, value in batch.items()
    }

    # Because masked_select flattens the output, we should recalculate the shape of non-padded entries
    shape_size = non_padded_mask.sum()  # Total number of non-padded entries

    # Now we need to reshape the tensors to the correct shape.
    # This shape will depend on the field e.g., (shape_size x feature_size)
    filtered_batch_reshaped = {
        key: value.view(shape_size, -1) if value.numel() > 0 else value
        for key, value in filtered_batch.items()
    }


# def preprocess_batch(batch, env, device, indices, sample_middle_state = True):
#     import numpy as np
#     from utils.kitchen_descriptions import LANGUAGE_DESCRIPTION, id2key
#     from utils.ball_descriptions import template_list
#     from utils.ball_descriptions import get_balls_description

#     description_indices = np.array(indices)[:, 1]
#     indices = np.array(indices)[:, 0]
#     obs = torch.from_numpy(batch["obs"][indices]).to(device).float()
#     next_obs = torch.from_numpy(batch["next_obs"][indices]).to(device).float()
#     valids = torch.from_numpy(batch["valid"][indices])
#     goals = batch["goals"][indices].astype(np.int16)

#     batch_ids = np.arange(len(obs))
#     traj_lengths = (valids.sum(1) - 1).reshape(-1).long()
#     next_obs = next_obs[batch_ids, traj_lengths]

#     sentences = []
#     if sample_middle_state:
#         if env == "kitchen":
#             tmp_valids = valids
#             tmp_obs = torch.zeros_like(next_obs)
#             for i in range(len(obs)):
#                 tmp_idx = tmp_valids[i].sum()
#                 tmp_idx = np.random.choice(int(tmp_idx))
#                 tmp_obs[i] = obs[i][tmp_idx]
#                 descriptions = LANGUAGE_DESCRIPTION[id2key[goals[i].item()]][description_indices[i]]
#                 sentences.append(descriptions)
#             obs = tmp_obs
#         elif env == "ball":
#             tmp_valids = valids
#             tmp_obs = torch.zeros_like(next_obs)
#             for i in range(len(obs)):
#                 tmp_idx = tmp_valids[i].sum()
#                 tmp_idx = np.random.choice(int(tmp_idx))
#                 tmp_obs[i] = obs[i][tmp_idx]
#                 descriptions = get_balls_description(goals[i], obs[i][tmp_idx].cpu().numpy(), next_obs[i].cpu().numpy(), description_indices[i])
#                 sentences.append(descriptions)
#             obs = tmp_obs
#         else:
#             raise NotImplementedError
#     return {
#         "obs": obs, "next_obs": next_obs, "sentences": sentences
#     }


# Your model
# model = ... (Define or load your model here)

# Assuming your model takes obs as input and possibly other batch as well
# Forward pass (make sure your model can handle the batch size, or split it into manageable batches if necessary)
# output = model(filtered_batch_reshaped['obs'], ...)
