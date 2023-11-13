# first download embeddings from here: https://github.com/ezhang7423/hulc-baseline/blob/main/dataset/download_lang_embeddings.sh

# typer app
import os
from typing_extensions import Annotated
from enum import Enum
import typer
from pathlib import Path
import torch
import numpy as np

app = typer.Typer()

# make the following into an enum compatible with typer:
# lang_all-distilroberta-v1/  lang_BERT/           lang_huggingface_distilroberta/  lang_paraphrase-MiniLM-L3-v2/ lang_all-mpnet-base-v2/     lang_clip_resnet50/  lang_huggingface_mpnet/

# class LanguageModelName(Enum):
#     distilroberta_v1 = "lang_all-distilroberta-v1"
#     BERT = "lang_BERT"
#     huggingface_distilroberta = "lang_huggingface_distilroberta"
#     paraphrase_MiniLM_L3_v2 = "lang_paraphrase-MiniLM-L3-v2"
#     mpnet_base_v2 = "lang_all-mpnet-base-v2"
#     clip_resnet50 = "lang_clip_resnet50"
#     huggingface_mpnet = "lang_huggingface_mpnet"
    
#     def __str__(self):
#         return self.value


model_choices = dict(
    BERT = "lang_BERT",
    distilroberta_v1 = "lang_all-distilroberta-v1",
    huggingface_distilroberta = "lang_huggingface_distilroberta",
    paraphrase_MiniLM_L3_v2 = "lang_paraphrase-MiniLM-L3-v2",
    mpnet_base_v2 = "lang_all-mpnet-base-v2",
    clip_resnet50 = "lang_clip_resnet50",
    huggingface_mpnet = "lang_huggingface_mpnet"
)
    
@app.command()
def main(data_dir: str, model_choice: str = 'clip_resnet50'):
    assert model_choice in model_choices.keys(), f"{model_choice} not in {model_choices.keys()}"
    
    ret = {}
    
    def extract_annotations(dir):
        annotations = np.load(dir / 'auto_lang_ann.npy', allow_pickle=True).item()
        unique_labels = set(annotations['language']['ann'])
        
        for i in unique_labels:
            idx = annotations['language']['ann'].index(i)
            ret[i] = torch.from_numpy(annotations['language']['emb'][idx])
    
    extract_annotations(Path(data_dir) / 'training' / model_choices[model_choice])
    extract_annotations(Path(data_dir) / 'validation' / model_choices[model_choice])
    
    from eztils import inspect
    inspect(ret)
    print(len(set(ret.keys())))
    return ret


if __name__ == "__main__":
    app()

# DATA_DIR = Path("/home/ezhang/Downloads/hulc-baseline/dataset")