from dataclasses import dataclass
from pathlib import Path
import copy
import datetime
import inspect

import torch
import tqdm
from lcd.utils.setup import set_seed
from lcd import DATA_PATH
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
from lcd.utils.eval import evaluate_policy, print_and_save
from lcd.apps.rollout import set_state

from accelerate import Accelerator
from torch.distributed.elastic.multiprocessing.errors import record
accelerator = Accelerator()
# device = 'cuda'
device = accelerator.device

def wlog(*args, **kwargs):
    if wandb.run is not None:
        wandb.log(*args, **kwargs)
    else:
        print(*args, **kwargs)

@dataclass
class eval_args:
    args: AttriDict = None
    eval_state: AttriDict = None
    model: torch.nn.Module = None
    skip_eval: bool = False
    num_sequences: int = 1000    

@record
def main(
    seed: int = 12,
    width: int = 4096,
    depth: int = 6,
    num_epochs: int = 10,
    num_steps: int = int(1e4),
    batch_size: int = 512,
    lr: float = 2e-5,
    use_wandb: bool = False,
    skip_eval: bool = False,
):    
    args, _, _, values = inspect.getargvalues(inspect.currentframe())
    args = AttriDict(
        {k: v for k, v in values.items() if k in args}
    )  #! is this function_locals only?
    
    if not skip_eval:
        eval_state = AttriDict()
        set_state(state=eval_state)
        eval_state.lang_embeddings = torch.load(DATA_PATH / "t5-v1_1-xxl_embeddings.pt")
    else:
        eval_state = None

    # setup
    set_seed(seed)
    exp_path = (
        DATA_PATH
        / "transformer-ablation"
        / datetime.datetime.now().strftime("%m-%d_%H:%M:%S.%f")
    )
    args.log_dir = exp_path
    exp_path.mkdir(parents=True)
    if use_wandb:
        wandb.init(
            project="vanilla-diffuser",
            entity="lang-diffusion",
            name=f"hulc-transformer-ablation",
            config=vars(args),
        )

    # model
    model = ActionTransformer(
        in_features=32, out_features=32, num_layers=depth, decoder_hidden_size=width
    )
    summary(model)

    # dataset
    d_args = utils.d_args()

    dataset_config = utils.Config(
        d_args.loader,
        savepath=(str(exp_path), "dataset_config.pkl"),
        horizon=d_args.horizon,
        normalizer=d_args.normalizer,
        preprocess_fns=d_args.preprocess_fns,
        use_padding=d_args.use_padding,
        max_path_length=d_args.max_path_length,
        frame_offset=d_args.frame_offset,
        lang_embeds=DATA_PATH / "t5-v1_1-xxl_embeddings.pt",
        task_to_ann=DATA_PATH / "annotations.json",
        buf=DATA_PATH / f"hulc-trajectories/{args.seed}_all_trajectories.pt",
        observation_dim=d_args.observation_dim,
        action_dim=d_args.action_dim,
    )
    dataset = dataset_config()
    dataloader = cycle(
        torch.utils.data.DataLoader(
            dataset,
            batch_size=int(
                batch_size / 4
            ),  # since we will slice the trajectory horizon into individual pieces, and the horizon length is 4
            num_workers=1,
            shuffle=True,
            pin_memory=True,
        )
    )
    

    eval_arg = eval_args(args, eval_state, model, skip_eval, 2)
    # validation evaluation is working
    evaluate(eval_arg)

    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
            
    model.train()
    for epoch in range(num_epochs):
        print('*'*50)
        print(f'{epoch=}')
        print('*'*50)
        torch.save(model, exp_path / f"model_{epoch}.pt")
        for i in tqdm.tqdm(range(num_steps)):
            batch: Batch = batch_to_device(next(dataloader)[0])
            traj = batch.trajectories.flatten(end_dim=1).to(device)
            pred = model.forward(
                traj[:, 32:], batch.conditions.repeat_interleave(4, dim=0)
            ).to(device)
            loss = ((pred - traj[:, :32]) ** 2).mean()
            accelerator.backward(loss)
            # loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if use_wandb:
                wlog({"train/loss": loss})
            elif not (i % 100):
                wlog({"train/loss": loss.item()})
                

        if not (epoch % 5):
            eval_arg.num_sequences = 10
            evaluate(eval_arg)
            

    eval_arg.num_sequences = 1000
    evaluate(eval_arg)


# %%


def evaluate(eval_arg: eval_args):
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


if __name__ == "__main__":
    
    main(
        width=16384,
        depth=24,
        # batch_size=72,
        # skip_eval=True,
        # lr=2e-5,
        use_wandb=True
    )
