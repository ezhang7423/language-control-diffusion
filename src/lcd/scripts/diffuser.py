import os

import torch

import lcd.utils as utils
import wandb
from lcd import DATA_PATH
from lcd.apps import rollout
from lcd.datasets.sequence import Batch
from lcd.utils.arrays import batch_to_device
from lcd.utils.training import cycle

script_dir = os.path.dirname(os.path.realpath(__file__))
# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#


class Parser(utils.Parser):
    config: str = "lcd.config.calvin"


args = Parser().parse_args("diffusion")
args.dim_mults = tuple(int(i) for i in args.dim_mults)

# -----------------------------------------------------------------------------#
# ---------------------------------- dataset ----------------------------------#
# -----------------------------------------------------------------------------#

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
)

# -----------------------------------------------------------------------------#
# -------------------------------- instantiate --------------------------------#
# -----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset)


# -----------------------------------------------------------------------------#
# ------------------------ test forward & backward pass -----------------------#
# -----------------------------------------------------------------------------#

utils.report_parameters(model)
print("Testing forward...", end=" ", flush=True)
dataloader = cycle(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0))
batch = batch_to_device(next(dataloader))
if isinstance(batch, Batch):
    loss, infos = diffusion.loss(*batch)
else:
    loss, infos = diffusion.loss(*batch[0], **batch[1])
loss.backward()
print("✓")


# -----------------------------------------------------------------------------#
# --------------------------------- main loop ---------------------------------#
# -----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

if args.wandb:
    wandb.init(
        project="vanilla-diffuser",
        entity="lang-diffusion",
        name=f"hulc-{args.wandb_name}",
        config=vars(args),
    )


def eval_model(num_evals, epoch=0):
    rollout.main(seed=args.seed, num_sequences=num_evals)
    dm_args = args.as_dict()
    dm_args["epoch"] = epoch
    rollout.lcd(dm__args=(diffusion, dm_args))


print("Testing evaluation...", end=" ", flush=True)
eval_model(
    num_evals=2,
)
print("✓")


for i in range(n_epochs):
    print(f"Epoch {i} / {n_epochs} | {args.savepath}")

    trainer.train(n_train_steps=args.n_steps_per_epoch)
    if not (i % args.eval_freq):
        eval_model(
            num_evals=args.n_evals_per_epoch,
            epoch=args.n_steps_per_epoch * (i + 1),
        )

eval_model(
    num_evals=1000,
    epoch=args.n_steps_per_epoch * (i + 1),
)
