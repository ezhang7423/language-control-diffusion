# %%
import open_clip
from torchinfo import summary
from lcd import DATA_PATH
import torch
import json

model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai')
tokenizer = open_clip.get_tokenizer('RN50')
NAME = "clip-rn50"


summary(model)


anns = sum(json.load(open(DATA_PATH /"annotations.json")).values(), [])
embeds = {}
with torch.no_grad(), torch.cuda.amp.autocast():
    embeddings = model.encode_text(tokenizer(anns))
    
for a, e in zip(anns, embeddings):
    if len(e.shape) == 1:
        e = e[None]
    embeds[a] = e.cpu()

print('Saving')    
save_path = DATA_PATH/ f"{NAME}_embeddings.pt"
torch.save(embeds, save_path)
print(f"Saved to {save_path}")