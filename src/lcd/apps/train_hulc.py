import os
import sys
from pathlib import Path

import typer
from loguru import logger

from lcd import HULC_PATH

py = sys.executable


def cache_shm_dataset(dataset):
    if not os.system(f"""tmux new-session -d -s calvin_cache_dataset"""):
        # only continue if the tmux session wasn't created before
        os.system(f"""tmux send-keys -t calvin_cache_dataset 'cd {HULC_PATH}' Enter""")
        os.system(
            f"""tmux send-keys -t calvin_cache_dataset '{py} hulc/training.py datamodule.root_data_dir={dataset} model=dummy logger=tb_logger trainer.gpus=1' Enter"""
        )
        logger.info(
            "View dataset caching progress with: tmux a -t calvin_cache_dataset"
        )
        logger.info(
            "Please wait until the dataset has completely finished loading, then run this command again. This tmux session should remain running indefinitely in the background, effectively as a daemon."
        )
        exit(0)


def main(
    ctx: typer.Context,
):
    """Train the original hulc model"""
    if ctx.args:
        args = " ".join(ctx.args)
    else:
        args = f" trainer.gpus=-1 datamodule.root_data_dir={HULC_PATH / 'dataset/task_D_D'} seed=12"

    # parse dataset
    dataset = None
    for arg in args.split(" "):
        split_arg = arg.split("=")
        if len(split_arg) == 2 and split_arg[0] == "datamodule.root_data_dir":
            dataset = split_arg[1]

    if dataset is None:
        logger.error(
            "Must specify the dataset directory in the args with datamodule.root_data_dir=/path/to/data"
        )
        raise ValueError

    # cache shm dataset
    cache_shm_dataset(dataset)

    cmd = f"{py} {HULC_PATH / 'hulc/training.py'} {args}"
    logger.info(f"Running: \n{cmd}")
    os.system(cmd)
