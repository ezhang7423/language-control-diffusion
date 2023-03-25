# %%
import os
from os.path import join as j

import torch

p = "/home/ubuntu/vanilla-hulc-larel-baseline/evaluation"
files = [j(p, f) for f in os.listdir(p) if f.startswith("gen")]
os.chdir(p)

# %%
"""
{seq_length: [ traj... ]}
traj = [subtask, subtask...] # total of seq_length subtasks

subtask: dict_keys(['states', 'actions', 'goal_image', 'goal_lang', 'goal_task'])

subtask.states = [ { dict_keys(['rgb_obs', 'robot_obs', 'depth_obs', 'robot_obs_raw', 'scene_obs']) } ...] # rgb is normalized already. 
epth_obs is empty
subtask.actions = tensor ([len(subtask), 1, 1, 7])
subtask.goal_image = { dict_keys(['rgb_obs', 'robot_obs', 'depth_obs', 'robot_obs_raw', 'scene_obs']) }
subtask.goal_task = 'rotate_pink_block_right'
subtask.goal_lang = ?
"""
# s = torch.load(j(p, 'gen_history_HULC_D_D_02-01-2023-09:17:46.pt'), map_location='cpu')
# len(s[1][0][0])

"""
You can generate each key state with +- some random epsilon for each index
You can use this "embeddings = np.load(Path(val_dataset_path) / lang_folder / "embeddings.npy", allow_pickle=True).item()" for embedding

"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from hulc.models.hulc import Hulc


def gen_annotations_embeddings():
    n = np.load(
        "/home/ubuntu/vanilla-hulc-icml-baseline/data/task_D_D/training/lang_paraphrase-MiniLM-L3-v2/auto_lang_ann.npy",
        allow_pickle=True,
    ).item()
    ann, tasks, emb = n["language"].values()
    annotations = defaultdict(list)
    embeddings = defaultdict(list)
    for i in range(len(ann)):
        if ann[i] not in annotations[tasks[i]]:
            annotations[tasks[i]].append(ann[i])
            embeddings[tasks[i]].append(emb[i])
    return annotations, embeddings


annotations, embeddings = gen_annotations_embeddings()
# model = Hulc.load_from_checkpoint(Path("/home/eddie/git/lad/vhulc/checkpoints/HULC_D_D/saved_models/HULC_D_D.ckpt"))
# model.freeze()


# %%
import random

import tqdm


def sample(task):
    anns = annotations[task]
    idx = random.randint(0, len(anns) - 1)
    return anns[idx], embeddings[task][idx][0]


# def gen_goal(states):
#     emb = model.perceptual_encoder(aggregate(states), {}, None)
#     return model.visual_goal(emb.squeeze(dim=1))


def aggregate(states):
    return {
        "rgb_static": torch.concat([v["rgb_obs"]["rgb_static"] for v in states]),
        "rgb_gripper": torch.concat([v["rgb_obs"]["rgb_gripper"] for v in states]),
    }


# %%

# ret = defaultdict(list)
# for f in tqdm.tqdm(files):
#     f = torch.load(f, map_location="cpu")
#     for seq_length in f:
#         for traj in f[seq_length]:
#             for subtask in traj:
#                 goal_space_states = gen_goal(subtask["states"])
#                 ret["states"].append(goal_space_states)
#                 ret["goal_task"].append(subtask["goal_task"])
#                 ann, lang = sample(subtask["goal_task"])
#                 ret["goal_lang"].append(lang)
#                 ret["goal_ann"].append(ann)

#     inter = {}
#     inter["states"] = np.array(ret["states"], dtype=object)
#     inter["goal_lang"] = torch.tensor(np.array(ret["goal_lang"]))
#     inter["goal_task"] = np.array(ret["goal_task"])
#     inter["goal_ann"] = np.array(ret["goal_ann"])

#     torch.save(inter, "all_trajectories_all_states.pt")


# ? AWS FROM SCRATCH
# %%
from collections import defaultdict

import tqdm

for seed in [12, 13, 42]:
    seed = str(seed)
    p = "/home/ubuntu/vanilla-hulc-larel-baseline/evaluation"
    files = [j(p, f) for f in os.listdir(p) if f.startswith(f"TG{seed}")]
    os.chdir(p)

    ret = defaultdict(list)
    for f in tqdm.tqdm(files):
        try:
            f = torch.load(f, map_location="cpu")
            for traj in f:
                for subtask in traj:
                    # goal_space_states = gen_goal(subtask["states"])
                    ret["states"].append(subtask["states"])
                    ret["goal_task"].append(subtask["goal_task"])
                    ann, lang = sample(subtask["goal_task"])
                    ret["goal_lang"].append(lang)
                    ret["goal_ann"].append(ann)

            inter = {}
            inter["states"] = np.array(ret["states"], dtype=object)
            inter["goal_lang"] = torch.tensor(np.array(ret["goal_lang"]))
            inter["goal_task"] = np.array(ret["goal_task"])
            inter["goal_ann"] = np.array(ret["goal_ann"])
        except Exception as e:
            print("*" * 50)
            print(f)
            print(e)
            print("*" * 50)

        torch.save(inter, f"{seed}_all_trajectories.pt")
