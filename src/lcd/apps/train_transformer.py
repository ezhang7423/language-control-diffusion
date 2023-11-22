import datetime
import inspect
from dataclasses import dataclass

import torch
import tqdm
from eztils.torch import get_best_cuda
from eztils.torch.wandb import log_wandb_distribution
from rich import print
from torch.distributed.elastic.multiprocessing.errors import record
from torchinfo import summary

import wandb
from lcd import DATA_PATH
from lcd.models.transformer import ActionTransformer
from lcd.utils.clevr import load_dataset
from lcd.utils.clevr.eval import DryEvalArgs, EvalArgs, evaluate, setup_env
from lcd.utils.config import AttriDict
from lcd.utils.setup import set_seed
from lcd.utils.training import cycle


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


def freeze(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = True


def main(
    seed: int = 12,
    width: int = 256,
    depth: int = 2,
    num_epochs: int = 10,
    num_steps: int = int(1e4),
    batch_size: int = 512,
    lr: float = 2e-5,
    use_wandb: bool = False,
    hierarchical: bool = False,
    low_model_path: str = str(DATA_PATH / "models" / "mlp-llp" / "model.pt"),
    skip_eval: bool = False,
):
    args, _, _, values = inspect.getargvalues(inspect.currentframe())
    args = AttriDict({k: v for k, v in values.items() if k in args})

    # setup
    set_seed(seed)
    exp_path = (
        DATA_PATH
        / "transformer-clevr"
        / datetime.datetime.now().strftime("%m-%d_%H:%M:%S.%f")
    )
    args.log_dir = exp_path
    exp_path.mkdir(parents=True)
    if use_wandb:
        wandb.init(
            project="vanilla-diffuser",
            entity="lang-diffusion",
            name="transformer-clevr-single",
            config=vars(args),
        )
    if hierarchical:
        low = torch.load(low_model_path)
        in_features = 16
        out_features = 16
    else:
        in_features = 10
        out_features = 40

    # model
    model = ActionTransformer(
        in_features=in_features,
        out_features=out_features,
        num_layers=depth,
        decoder_hidden_size=256,
        num_heads=2,
    )

    print(model)
    summary(model)
    # dataset
    train_dataset, val_dataset = load_dataset(
        clevr=True,
        exp_path=exp_path,
        upscaled_state_dim=0,
        upscale=False,
        shuffle=True,
        encoder=low if hierarchical else None,
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

    # eval_arg = EvalArgs(
    #     model=model,
    #     skip_eval=skip_eval,
    #     env=setup_env(max_episode_steps=50) if not skip_eval else None,
    # )
    # validation evaluation is working
    torch.save(model, exp_path / f"model_initial.pt")
    ret = evaluate(
        DryEvalArgs(
            high_model_path=str(exp_path / f"model_initial.pt"),
            low_model_path=low_model_path,
            transformer=True,
            only_hlp=not hierarchical,
        ),
        eval_all=False,
        skip_eval=False,
        num_processes=1,
    )
    wlog(ret)

    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = "cuda:" + str(
        get_best_cuda()
    )  # TODO make get_best_cuda return string directly
    model.to(device)
    model.train()

    from torch.nn import functional as F

    for epoch in range(num_epochs):
        # for epoch in range(1):
        print("*" * 50)
        print(f"{epoch=}")
        print("*" * 50)
        model_path = exp_path / f"model_{epoch}.pt"
        torch.save(model, model_path)
        torch.save(optimizer, exp_path / f"optimizer_{epoch}.pt")
        # if not (epoch % 5):
        wlog(
            evaluate(
                DryEvalArgs(
                    high_model_path=str(model_path),
                    num_sequences=100,
                    low_model_path=low_model_path,
                    transformer=True,
                    only_hlp=not hierarchical,
                ),
                skip_eval=False,
                num_processes=1,
            )
        )

        for i in tqdm.tqdm(range(num_steps)):
            # for i in range(100):
            def get_loss(dataloader, log_val=False):
                # batch = preprocess_batch(next(dataloader), device)
                extra_stats = {}
                batch = next(dataloader).to(device).float()

                #! change this train part
                pred = model.forward(
                    # torch.concat((batch["obs"], batch["next_obs"]), dim=-1)
                    batch["obs"],
                    torch.stack(
                        [
                            embeds[str(x)]
                            for x in batch["obs_goal"][:, -3:].int().tolist()
                        ]
                    ).to(batch["obs"]),
                ).to(device)

                # pred = model.forward(
                #     traj[:, 32:], batch.conditions.repeat_interleave(4, dim=0)
                # ).to(device)
                if hierarchical:
                    loss = F.mse_loss(pred, batch["next_obs"])
                else:
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

    wlog(
        evaluate(
            DryEvalArgs(
                high_model_path=str(model_path),
                low_model_path=low_model_path,
                transformer=True,
                only_hlp=not hierarchical,
            ),
            eval_all=True,
        )
    )


embeds = torch.load(DATA_PATH / "clevr_direct_embeddings.pt")


def preprocess_obs(obs):
    return obs[:, :-3], embeds[str(obs[0, -3:].int().tolist())]


if __name__ == "__main__":
    import typer

    app = typer.Typer(pretty_exceptions_show_locals=False)

    app.command()(main)
    app()
