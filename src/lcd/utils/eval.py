# originally taken from https://github.com/lukashermann/hulc/blob/fb14d5461ae54f919d52c0c30131b38f806ef8db/hulc/evaluation/evaluate_policy.py and adapted for hierarchical imitation learning and data collection
import gc
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from hulc.evaluation.utils import get_env_state_for_initial_condition, join_vis_lang
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

import lcd
from lcd import DATA_PATH, HULC_PATH, REPO_PATH
from lcd.utils.serialization import load_config


class DiffusionModelWrapper(torch.nn.Module):
    def __init__(
        self, device="cpu", model_path__epoch=(None, None), model__args=(None, None)
    ) -> None:
        """
        Wrapper for the LCD diffusion model to use during evaluation.
        Can either take in model directly through model__args or load from the filesystem with model_path__epoch
        """
        super().__init__()
        model_path, epoch, model, args = (*model_path__epoch, *model__args)

        assert model_path or model, "Need to specify either model_path or model"

        if model_path:
            sys.modules["diffuser"] = lcd
            model_config = load_config(model_path, "model_config.pkl")
            diffusion_config = load_config(model_path, "diffusion_config.pkl")
            init_hulc_goal_state_dicts = torch.load(
                os.path.join(model_path, f"state_{epoch}.pt"), map_location="cpu"
            )
            model = model_config()
            diffusion = diffusion_config(model)
            diffusion.load_state_dict(init_hulc_goal_state_dicts["ema"])
            self.model = diffusion
            self.args = json.load(open(f"{model_path}/args.json"))
        else:
            self.model = model
            self.args = args

        self.model.to(device)

    def forward(self, cond, inpaint):
        samples = self.model.conditional_sample(
            cond, horizon=1, inpaint={0: inpaint}
        ).trajectories
        return samples[:, :, :32].squeeze(dim=1)


def get_sequences(args, regenerate=False):
    if regenerate:
        from hulc.evaluation.multistep_sequences import get_sequences

        # this takes a few minutes
        return get_sequences(args.num_sequences)
    else:
        return torch.load(DATA_PATH / "default_1000_sequences.pt")[: args.num_sequences]


def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def get_log_dir(log_dir):
    log_dir = Path(log_dir) if log_dir is not None else REPO_PATH / "results"
    if not log_dir.exists():
        log_dir.mkdir()
    print(f"logging to {log_dir}")
    return log_dir


def print_and_save(results, args, histories, model_id):
    def get_task_success_rate(results, sequences):
        cnt_success = Counter()
        cnt_fail = Counter()

        for result, (_, sequence) in zip(results, sequences):
            for successful_tasks in sequence[:result]:
                cnt_success[successful_tasks] += 1
            if result < len(sequence):
                failed_task = sequence[result]
                cnt_fail[failed_task] += 1

        total = cnt_success + cnt_fail
        task_info = {}
        for task in total:
            task_info[task] = {"success": cnt_success[task], "total": total[task]}
            print(
                f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%"
            )
        return task_info

    log_dir = get_log_dir(args.log_dir)

    sequences = get_sequences(args)
    if args.generate:
        Path(log_dir / "rollouts").mkdir(
            exist_ok=True, parents=True
        )  # create rollouts folder if it doesn't exist
        torch.save(
            histories,
            log_dir
            / f"rollouts/{model_id}_gen_history_{datetime.now().strftime('%d-%m-%Y-%H:%M:%S')}.pt",
        )

    current_data = {}
    # epoch = checkpoint.stem
    print(f"Results for Model {model_id}:")
    avg_seq_len = np.mean(results)
    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
    print(f"Average successful sequence length: {avg_seq_len}")
    print("Success rates for i instructions in a row:")
    for i, sr in chain_sr.items():
        print(f"{i}: {sr * 100:.1f}%")

    data = {
        "avg_seq_len": avg_seq_len,
        "chain_sr": chain_sr,
        "task_info": get_task_success_rate(results, sequences),
    }

    current_data[model_id] = data
    previous_data = {}

    try:
        with open(log_dir / "results.json") as file:
            previous_data = json.load(file)
    except FileNotFoundError:
        pass
    json_data = {**previous_data, **current_data}
    with open(log_dir / "results.json", "w") as file:
        json.dump(json_data, file)
    return data


def evaluate_policy(state, args):
    conf_dir = HULC_PATH / "conf"
    task_cfg = OmegaConf.load(
        conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(
        conf_dir / "annotations/new_playtable_validation.yaml"
    )

    eval_sequences = get_sequences(args)
    if not args.debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    results = []
    histories = []
    for initial_state, eval_sequence in eval_sequences:
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        state.env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        success_counter = 0
        history = []

        if args.debug:
            time.sleep(1)
            print()
            print()
            print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
            print("Subtask: ", end="")
        for subtask in eval_sequence:
            success, info = rollout(
                state.env,
                state.model,
                task_oracle,
                args,
                subtask,
                state.lang_embeddings,
                val_annotations,
            )
            if success:
                success_counter += 1
                history.append(info)
            else:
                break

        histories.append(history)
        results.append(success_counter)
        if not args.debug:
            eval_sequences.set_description(
                " ".join(
                    [
                        f"{i + 1}/5 : {v * 100:.1f}% |"
                        for i, v in enumerate(count_success(results))
                    ]
                )
                + "|"
            )

    return results, histories


def rollout(env, model, task_oracle, args, subtask, lang_embeddings, val_annotations):
    if args.debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    if args.generate:
        states = []
        actions = []
        scene_info = []

    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]

    if args.dm is not None:
        model.replan_freq = args.subgoal_interval
        current_state = model.get_pp_plan_vision(obs, obs)[-1]
        lang_goal = lang_embeddings[lang_annotation].to(args.device)
        goal = {"lang": args.dm(lang_goal[None], current_state)}
    else:
        goal = lang_embeddings.get_lang_goal(lang_annotation)
        plan, latent_goal = model.get_pp_plan_lang(obs, goal)

    model.reset()
    start_info = current_info = env.get_info()
    success_step = 0
    success = False

    for step in range(args.ep_len):
        if (
            success_step and step == success_step + 4
        ):  # record four states past the successful state
            break
        if args.dm is not None:
            if not (step % args.subgoal_interval):
                current_state = (
                    model.visual_goal(
                        model.perceptual_encoder(obs["rgb_obs"], {}, None)
                    )
                    .squeeze()
                    .cpu()
                )
                goal = {"lang": args.dm(lang_goal[None], current_state)}

        action = model.step(obs, goal, direct=args.dm is not None)

        if args.generate:
            states.append(
                model.visual_goal(model.perceptual_encoder(obs["rgb_obs"], {}, None))
                .squeeze()
                .cpu()
            )
            scene_info.append(current_info)
            actions.append(action.squeeze().cpu())

        obs, _, _, current_info = env.step(action)

        if args.debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(
            start_info, current_info, {subtask}
        )
        if len(current_task_info) > 0 and not success:
            if args.debug:
                logger.info("<green>Success</green>")
            success = True
            success_step = step

    if args.debug and not success:
        logger.info("<red>Fail</red>")

    if not args.generate:
        return success, {}
    else:
        gc.collect()
        torch.cuda.empty_cache()
        return success, {
            "states": torch.stack(states).cpu().detach(),
            "actions": torch.stack(actions).cpu().detach(),
            "goal_lang": goal["lang"].cpu().detach().squeeze(),
            "goal_task": subtask,
            "scene_info": scene_info,
        }
