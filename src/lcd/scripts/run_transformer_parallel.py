import itertools
import os
import time

seeds = [0, 12, 42]
sizes = [(4096, 4)]
lr = [2e-4, 4e-4]

for i, (l, seed, size) in enumerate(itertools.product(lr, seeds, sizes)):
    gpu_id = (i % 8)
    os.system(f'CUDA_VISIBLE_DEVICES={gpu_id} lcd train_transformer --lr {l} --seed {seed} --use-wandb --width {size[0]} --depth {size[1]} > {i}.log 2>&1 &')
    time.sleep(1.1)
