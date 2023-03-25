from typing import List

import json

import torch
from einops import rearrange
from transformers import T5EncoderModel, T5Tokenizer

DEFAULT_T5_NAME = "google/t5-v1_1-base"
T5_CONFIGS = {}
MAX_LENGTH = 256


# taken from https://github.com/lucidrains/imagen-pytorch/blob/35f24ea102ab1d71da7df3c8a650c4fe712d2a9c/imagen_pytorch/t5.py#L107
def t5_encode_text(texts: List[str], name=DEFAULT_T5_NAME, return_attn_mask=False):
    token_ids, attn_mask = t5_tokenize(texts, name=name)
    encoded_text = t5_encode_tokenized_text(token_ids, attn_mask=attn_mask, name=name)

    if return_attn_mask:
        attn_mask = attn_mask.bool()
        return encoded_text, attn_mask

    return encoded_text


def exists(val):
    return val is not None


def get_tokenizer(name):
    tokenizer = T5Tokenizer.from_pretrained(name, model_max_length=MAX_LENGTH)
    return tokenizer


def get_model(name):
    model = T5EncoderModel.from_pretrained(name)
    return model


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def get_model_and_tokenizer(name):
    global T5_CONFIGS

    if name not in T5_CONFIGS:
        T5_CONFIGS[name] = dict()
    if "model" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["model"] = get_model(name)
    if "tokenizer" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["tokenizer"] = get_tokenizer(name)

    return T5_CONFIGS[name]["model"], T5_CONFIGS[name]["tokenizer"]


def t5_tokenize(texts: List[str], name=DEFAULT_T5_NAME):
    t5, tokenizer = get_model_and_tokenizer(name)

    if torch.cuda.is_available():
        t5 = t5.cuda()

    device = next(t5.parameters()).device

    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors="pt",
        padding="longest",
        max_length=MAX_LENGTH,
        truncation=True,
    )

    input_ids = encoded.input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)
    return input_ids, attn_mask


def t5_encode_tokenized_text(
    token_ids, attn_mask=None, pad_id=None, name=DEFAULT_T5_NAME
):
    assert exists(attn_mask) or exists(pad_id)
    t5, _ = get_model_and_tokenizer(name)

    attn_mask = default(attn_mask, lambda: (token_ids != pad_id).long())

    t5.eval()

    with torch.no_grad():
        output = t5(input_ids=token_ids, attention_mask=attn_mask)
        encoded_text = output.last_hidden_state.detach()

    attn_mask = attn_mask.bool()

    encoded_text = encoded_text.masked_fill(
        ~rearrange(attn_mask, "... -> ... 1"), 0.0
    )  # just force all embeddings that is padding to be equal to 0.
    return encoded_text


NAME = "t5-v1_1-xxl"
anns = sum(json.load(open("/data2/eddie/calvin/annotations.json")).values(), [])
embeds = {}
embeddings = t5_encode_text(anns, name=f"google/{NAME}")
for a, e in zip(anns, embeddings):
    embeds[a] = e.cpu()
torch.save(embeds, f"{NAME}_embeddings.pt")
