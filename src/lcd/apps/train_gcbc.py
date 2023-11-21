# %%
import datetime
import inspect

import torch
import tqdm
from eztils.torch import get_best_cuda
from eztils.torch.wandb import log_wandb_distribution
from rich import print
from torchinfo import summary

import wandb
from lcd import DATA_PATH
from lcd.models.mlp import ForwardModel
from lcd.utils.clevr import load_dataset
from lcd.utils.clevr.eval import EvalArgs, evaluate, setup_env
from lcd.utils.config import AttriDict
from lcd.utils.setup import set_seed
from lcd.utils.training import cycle


def wlog(*args, **kwargs):
    if wandb.run is not None:
        wandb.log(*args, **kwargs)
    else:
        print(*args, **kwargs)


def main(
    seed: int = 12,
    width: int = 256,
    upscaled_state_dim: int = 2048,
    upscale: bool = False,
    depth: int = 3,
    num_epochs: int = 100,
    num_steps: int = int(1e4),
    batch_size: int = 512,
    lr: float = 2e-4,
    diffusion_dim: int = 16,
    use_wandb: bool = False,
    skip_eval: bool = True,
    clevr: bool = True,
    weight_decay: float = 1e-4,
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
    if use_wandb:
        wandb.init(
            project="vanilla-diffuser",
            entity="lang-diffusion",
            name=f"{env_name}-ce-post-activation-weight-decay",
            config=vars(args),
        )
    print("Saving to", exp_path)

    if not upscale:
        upscaled_state_dim = 10
    # model
    final_dim = 40 if clevr else None
    num_units = [width] * depth
    num_units[-1] = diffusion_dim
    model = ForwardModel(
        in_dim=upscaled_state_dim,
        final_dim=final_dim,
        diffusion_dim=diffusion_dim,
        num_units=num_units,
    )
    # model = torch.load("/home/ubuntu/talar/lcd-iclr24-clevr/submodules/data/clevr/11-19_18:42:51.252737/model_99.pt",    )
    print(model)
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
    eval_arg = EvalArgs(
        args=args,
        model=model,
        skip_eval=skip_eval,
        env=setup_env() if not skip_eval else None,
    )

    # test evaluation is working
    evaluate(eval_arg)

    # train
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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
                    # torch.concat((batch["obs"], batch["next_obs"]), dim=-1)
                    batch["obs"],
                    batch["next_obs"],
                ).to(device)

                act_int = batch["actions"].to(torch.int64)
                loss = F.cross_entropy(pred, act_int)
                if log_val:
                    correct = pred.argmax(dim=1) == act_int
                    extra_stats["eval/acc"] = sum(correct) / len(correct)
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
