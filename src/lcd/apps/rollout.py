from pathlib import Path

import torch
import typer
from hulc.evaluation.utils import get_default_model_and_env
from loguru import logger

from lcd import DATA_PATH, HULC_PATH
from lcd.utils.config import AttriDict
from lcd.utils.eval import DiffusionModelWrapper, evaluate_policy, print_and_save

_app_ = app = typer.Typer(name="lcd")

args = AttriDict()
state = AttriDict()


def eval_pipeline():
    logger.debug(f"{args=}")
    results, histories = evaluate_policy(state, args)
    model_id = f"{args.train_folder=}_{args.seed=}"
    if args.diffusion_path is not None:
        model_id += f"_{args.diffusion_path=}_{args.diffusion_epoch=}"
    print_and_save(results, args, histories, model_id)


@app.callback()
def main(
    dataset_path: str = str(HULC_PATH / "dataset/task_D_D"),
    train_folder: str = str(DATA_PATH / "hulc-baselines-30"),
    seed: int = 12,
    debug: bool = False,
    log_dir: str = None,
    device: int = 0,
    num_sequences: int = 1000,
):
    """
    Rollout in the environment for evaluation or dataset collection
    """
    ep_len = 360
    args.update(locals())

    # *******
    (
        state.model,
        state.env,
        _,
        state.lang_embeddings,
    ) = get_default_model_and_env(
        args.train_folder,
        args.dataset_path,
        Path(args.train_folder) / f"saved_models/seed={args.seed}.ckpt",
        env=None,
        lang_embeddings=None,
        device_id=args.device,
    )


@app.command()
def lcd(
    diffusion_path: str = DATA_PATH / "lcd-seeds/seed-12",
    diffusion_epoch: str = 250_000,
    subgoal_interval: int = 16,
    dm__args: str = None,
):
    """
    Rollout with LCD
    """
    args.update(locals())

    # *******
    if dm__args:
        args.dm = DiffusionModelWrapper(
            device=f"cuda:{args.device}",
            model__args=dm__args,
        )
        args.diffusion_path = dm__args[1]["savepath"]
        args.diffusion_epoch = dm__args[1]["epoch"]
    else:
        args.dm = DiffusionModelWrapper(
            device=f"cuda:{args.device}",
            model_path__epoch=(args.diffusion_path, args.diffusion_epoch),
        )

    state.lang_embeddings = torch.load(DATA_PATH / "t5-v1_1-xxl_embeddings.pt")

    eval_pipeline()


@app.command()
def hulc():
    """
    Rollout with HULC
    """
    eval_pipeline()


@app.command()
def generate():
    """
    Generate an on-policy dataset for training a high level policy (e.g. LCD)
    """
    args.generate = True
    eval_pipeline()
