import argparse
import os
import sys

import torch

import lcd.utils as utils
import wandb
from lcd import DATA_PATH
from lcd.apps import rollout
from lcd.datasets.sequence import Batch
from lcd.utils.arrays import batch_to_device
from lcd.utils.clevr import load_dataset
from lcd.utils.setup import set_seed
from lcd.utils.training import cycle

script_dir = os.path.dirname(os.path.realpath(__file__))
# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#

# Create an ArgumentParser object
parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument("--benchmark", type=str, default="clevr", help="which benchmark?")
# Parse the command-line arguments
args, unknown_args = parser.parse_known_args()
if args.benchmark not in ["clevr", "calvin"]:
    raise NotImplementedError(f"benchmark {args.benchmark} not implemented")


class Parser(utils.Parser):
    config: str = f"lcd.config.{args.benchmark}"


args = Parser().parse_args("diffusion")
args.dim_mults = tuple(int(i) for i in args.dim_mults)

set_seed(args.seed)

# -----------------------------------------------------------------------------#
# ---------------------------------- dataset ----------------------------------#
# -----------------------------------------------------------------------------#
if args.end2end:
    args.loader = "datasets.ClevrEnd2EndDataset"
    args.observation_dim = 10
    args.action_dim = 40

if args.benchmark == "clevr":
    print("loading dataset...")
    train, val = load_dataset(True, exp_path=None)
    print("done loading!")

    dataset_config = utils.Config(
        args.loader,
        savepath=(args.savepath, "dataset_config.pkl"),
        horizon=args.horizon,
        normalizer=args.normalizer,
        preprocess_fns=args.preprocess_fns,
        use_padding=args.use_padding,
        max_path_length=args.max_path_length,
        frame_offset=args.frame_offset,
        encoder_path=args.llp_path,
        lang_embeds=DATA_PATH / "clevr_direct_embeddings.pt",
        observation_dim=args.observation_dim,
        action_dim=args.action_dim,
    )

    dataset = dataset_config(buf=train)
    val_dataset = dataset_config(buf=val)
else:
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
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim


# -----------------------------------------------------------------------------#
# ------------------------------ model & trainer ------------------------------#
# -----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, "model_config.pkl"),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    dim=args.model_dim,
    attention=args.attention,
    device=args.device,
    downsample=args.downsample,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, "diffusion_config.pkl"),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, "trainer_config.pkl"),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    dev=args.device,
)

# -----------------------------------------------------------------------------#
# -------------------------------- instantiate --------------------------------#
# -----------------------------------------------------------------------------#
model = model_config()
diffusion = diffusion_config(model)

if args.benchmark == "clevr":
    trainer = trainer_config(diffusion, dataset, val_dataset)
else:
    trainer = trainer_config(diffusion, dataset)


# -----------------------------------------------------------------------------#
# ------------------------ test forward & backward pass -----------------------#
# -----------------------------------------------------------------------------#

utils.report_parameters(model)
print("Testing forward...", end=" ", flush=True)
dataloader = cycle(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0))
batch = batch_to_device(next(dataloader), device=args.device)
if isinstance(batch, Batch):
    loss, infos = diffusion.loss(*batch)
else:
    loss, infos = diffusion.loss(*batch[0], **batch[1])
loss.backward()
print("✓")


# -----------------------------------------------------------------------------#
# --------------------------------- main loop ---------------------------------#
# -----------------------------------------------------------------------------#
if args.wandb:
    wandb.init(
        project="vanilla-diffuser",
        entity="lang-diffusion",
        name=f"hulc-{args.wandb_name}",
        config=vars(args),
    )


n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)


def eval_model(num_evals, epoch=0):
    trainer.save(epoch)
    if args.benchmark == "clevr":
        from lcd.utils.clevr.eval import DryEvalArgs, evaluate

        if num_evals < 25:
            num_processes = 5
            num_sequences = max(1, num_evals // num_processes)
        else:
            num_processes = 25
            num_sequences = num_evals // num_processes

        avg, num_evals = evaluate(
            dry_eval_args=DryEvalArgs(
                low_model_path=args.llp_path,
                high_model_path=os.path.join(args.savepath, f"model_{epoch}.pt"),
                num_sequences=num_sequences,
                only_hlp=True if args.end2end else False,
            ),
            num_processes=num_processes,
        )
        if args.wandb:
            wandb.log({"eval/sr": avg, "eval/num_evals": num_evals})
    else:
        rollout.main(seed=args.seed, num_sequences=num_evals)
        dm_args = args.as_dict()
        dm_args["epoch"] = epoch
        rollout.lcd(dm__args=(diffusion, dm_args))


print("Testing evaluation...", end=" ", flush=True)
evaluation = eval_model(num_evals=100, epoch="epoch_0")
print(evaluation)
print("✓")

if args.wandb:
    wandb.log(
        {
            "buffer_histogram": wandb.Histogram(
                dataset.buf["obs"][:50].flatten().tolist()
            ),
        }
    )


for i in range(n_epochs):
    print(f"Epoch {i} / {n_epochs} | {args.savepath}")

    trainer.train(n_train_steps=args.n_steps_per_epoch)
    if not (i % args.eval_freq):
        eval_model(
            num_evals=100,
            epoch=f"epoch_{i+1}",
        )

eval_model(
    num_evals=100,
    epoch=f"epoch_{i+1}",
)
