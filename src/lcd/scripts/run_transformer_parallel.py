import itertools
import os


seeds = [0, 12, 42]
sizes = [(2048, 4), (4096, 6), (8192, 24)]


for i, (seed, size) in enumerate(itertools.product(seeds, sizes)):
    gpu_id = (i % 8) + 3
    os.system(f'CUDA_VISIBLE_DEVICES={gpu_id} lcd train_transformer --seed {seed} --use-wandb --width {size[0]} --depth {size[1]} | tee {i}.log &')
