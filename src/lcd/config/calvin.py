from lcd.utils import watch

# ------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ("prefix", ""),
    # ('horizon', 'H'),
    ("n_diffusion_steps", "T"),
    ## value kwargs
    ("seed", "S"),
]

logbase = "logs"

base = {
    "diffusion": {
        ## model
        "model": "models.TemporalUnet",
        "diffusion": "models.GaussianDiffusion",
        "horizon": 4,  # 1,
        "n_diffusion_steps": 20,
        "action_weight": 100,
        "loss_weights": None,
        "loss_discount": 1,
        "predict_epsilon": False,
        "dim_mults": (1, 4, 8),  # (1, 2, 4, 8),
        "attention": True,  # False
        "renderer": "utils.MuJoCoRenderer",
        "downsample": False,  # True
        "model_dim": 64,  # 128,
        ## dataset
        "loader": "datasets.HulcDataset",  # 'datasets.HulcDataset',
        "frame_offset": 0,
        "normalizer": "GaussianNormalizer",
        "preprocess_fns": [],
        "clip_denoised": False,
        "use_padding": True,
        "max_path_length": 1000,
        "observation_dim": 32,
        "action_dim": 32,
        ## serialization
        "logbase": logbase,
        "prefix": "diffusion/defaults",
        "exp_name": watch(args_to_watch),
        ## training
        "wandb": False,  # true
        "wandb_name": "default",
        "wandb_project": "language-control-diffusion",
        # "wandb_entity": "<REPLACE WITH YOUR WANDB ORG>",  #!! Change me
        "wandb_entity": "lang-diffusion",  #!! Change me
        "n_steps_per_epoch": 10000,
        "n_evals_per_epoch": 10,
        "eval_freq": 10,
        "loss_type": "l2",
        "n_train_steps": 3e5,
        "batch_size": 512,
        "learning_rate": 2e-4,
        "gradient_accumulate_every": 1,
        "ema_decay": 0.995,
        "save_freq": 10000,
        "sample_freq": 1000,
        "n_saves": 5,
        "save_parallel": False,
        "n_reference": 8,
        "bucket": None,
        "device": "cuda",
        "seed": 0,
    }
}
