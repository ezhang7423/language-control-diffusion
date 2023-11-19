from eztils.default.run_parallel import run_parallel, BaseHyperParameters


class HyperParams(BaseHyperParameters):
    width = [64, 128, 256]
    depth = [2, 3, 4]
    lr = [2e-3, 2e-4, 2e-5]
    use_wandb = [True]
    # batch_size = [64, 128, 512, 1024]


run_parallel(
    HyperParams, 
    base_cmd='lcd train_gcbc',
    data_path='runs',
    sleep_time=10,
)