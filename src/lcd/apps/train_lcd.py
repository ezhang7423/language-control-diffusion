import os
import sys
from pathlib import Path

import typer
from loguru import logger

from lcd import HULC_PATH, REPO_PATH

py = sys.executable


def main(
    ctx: typer.Context,
):
    """Train the original hulc model"""
    if ctx.args:
        args = " ".join(ctx.args)
    else:
        args = f" --seed 12 --wandb True"

    cmd = f"{py} {REPO_PATH / 'src/lcd/scripts/diffuser.py'} {args}"
    logger.info(f"Running: \n{cmd}")
    os.system(cmd)
