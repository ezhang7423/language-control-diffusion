import datetime
import os
import sys
from pathlib import Path

import torch

import typer
from loguru import logger

from lcd import HULC_PATH, REPO_PATH, DATA_PATH
from lcd.utils.setup import set_seed
py = sys.executable


def main
    ctx: typer.Context,
):
    """Train the original hulc model"""
    if ctx.args:
        args = " ".join(ctx.args)
    else:
        args = f" --seed 12 --wandb True"
    if "--model transformer" in args:
        train_transformer_wrapper(ctx)
    
    cmd = f"{py} {REPO_PATH / 'src/lcd/scripts/diffuser.py'} {args}"
    logger.info(f"Running: \n{cmd}")
    os.system(cmd)

# -----------------------------------------------------------------------------#
# ----------------------------------- transformer------------------------------#
# -----------------------------------------------------------------------------#

import os

import torch

import lcd.utils as utils
import wandb
from lcd import DATA_PATH
from lcd.apps import rollout
from lcd.datasets.sequence import Batch
from lcd.utils.arrays import batch_to_device
from lcd.utils.training import cycle
from lcd.models.transformer import ActionTransformer, TransformerEvaluationWrapper
from torchinfo import summary
from lcd.utils.config import AttriDict    

class Parser(utils.Parser):
    config: str = "lcd.config.calvin"


def train_transformer_wrapper(ctx: typer.Context):
    print(ctx)
    seed = 0
    train_transformer(seed)

def wlog(*args, **kwargs):
    if wandb.run is not None:
        wandb.log(*args, **kwargs)

def train_transformer(seed: int, width=2048, depth=2, num_epochs=10, num_steps=int(1e4), batch_size=512, lr=2e-5, wandb=False):
    args =  AttriDict(locals()) #! is this function_locals only? 

    # setup    
    set_seed(seed)
    exp_path = DATA_PATH / 'transformer-ablation' / datetime.datetime.now().strftime("%m-%d_%H:%M:%S")
    exp_path.mkdir(parents=True)
    if wandb:    
        wandb.init(
        project="vanilla-diffuser",
        entity="lang-diffusion",
        name=f"hulc-transformer-ablation",
        config=vars(args),
        )

    # model
    model = ActionTransformer(in_features=32, out_features=32, num_layers=depth, decoder_hidden_size=width)
    summary(model)

    # dataset    
    args = Parser().parse_args("diffusion")
    dataset_config = utils.Config(
        args.loader,
        savepath=(args.savepath, "dataset_config.pkl"),
        horizon=args.horizon,
        normalizer=args.normalizer,
        preprocess_fns=args.preprocess_fns,
        use_padding=args.use_padding,
        max_path_length=args.max_path_length,
        frame_offset=args.frame_offset,
        lang_embeds=DATA_PATH / "t5-v1_1-xxl_embeddings.pt",
        task_to_ann=DATA_PATH / "annotations.json",
        buf=DATA_PATH / f"hulc-trajectories/{args.seed}_all_trajectories.pt",
        observation_dim=args.observation_dim,
        action_dim=args.action_dim,
    )
    dataset = dataset_config()
    dataloader = cycle(
            torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=1,
                shuffle=True,
                pin_memory=True,
            )
        )
    
    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        torch.save(exp_path / f'model_{epoch}.pt', model)
        for i in range(num_steps):
            batch: Batch = next(dataloader)[0]
            pred = model.forward(batch.trajectories, batch.conditions)
            loss = ((pred - batch.trajectories) **2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            wlog({'train/loss': loss})
            
    
    # run evaluation
    from lcd.utils.eval import DiffusionModelWrapper, evaluate_policy, print_and_save    
    from lcd.apps.rollout import set_state
    
    state = AttriDict()
    set_state(state=state)
    state.lang_embeddings = torch.load(DATA_PATH / "t5-v1_1-xxl_embeddings.pt")
    results, histories = evaluate_policy(state, args)    
    model_id = f"transformer-ablation-{args}"    
    args.dm = TransformerEvaluationWrapper(model)
    
    eval_results = print_and_save(results, args, histories, model_id)
    wlog({f'eval/{k}': v for k,v in eval_results.items()})
    

if __name__ == '__main__':
    train_transformer(seed=0)

