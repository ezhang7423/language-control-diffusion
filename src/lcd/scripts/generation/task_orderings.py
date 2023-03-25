from collections import defaultdict

import torch
from hulc.evaluation.multistep_sequences import get_sequences

seq = get_sequences(num_sequences=10000, num_workers=64)

first_task = [s[1][0] for s in seq]
indices = defaultdict(list)
for i, s in enumerate(seq):
    indices[s[1][0]].append(i)

first_indices = list(indices.keys())
for i, s in enumerate(seq):
    if s[1][1] not in first_indices:
        indices[s[1][1]].append(i)
    if s[1][2] == "unstack_block":
        indices[s[1][2]].append(i)

info = (indices, seq)
torch.save(info, "(indices,seq).pt")
