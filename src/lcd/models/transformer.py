import torch
from torch import nn
from typing import Tuple


class ActionTransformer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        decoder_normalize: bool = True,
        num_heads: int = 8,
        num_layers: int = 4,
        decoder_hidden_size: int = 2048,
        lang_features: int = 4096,
        fc_hidden_size: int = 4096,
        max_position_embeddings: int = 16,
        dropout_p: bool = 0.1,
        position_embedding: bool = True,
    ):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_position_embeddings = max_position_embeddings

        self.hidden_size = fc_hidden_size
        self.position_embedding = position_embedding
        self.decoder_normalize = decoder_normalize

        self.padding = False
        self.lang_projection = nn.Linear(lang_features, in_features)
        mod = self.in_features % num_heads
        if mod != 0:
            print(f"Padding for Num of Heads : {num_heads}")
            self.padding = True
            self.pad = num_heads - mod
            self.in_features += self.pad
            
        self.position_embeddings = nn.Embedding(max_position_embeddings, self.in_features)
        decoder_layer = nn.TransformerDecoderLayer(
            self.in_features, num_heads, activation='gelu', dim_feedforward=decoder_hidden_size, dropout=dropout_p, batch_first=True
        )
        decoder_norm = nn.LayerNorm(self.in_features) if decoder_normalize else None
        self.dropout = nn.Dropout(p=dropout_p)
        self.cross_attention = nn.TransformerDecoder(decoder_layer, num_layers=num_layers, norm=decoder_norm)
        self.fc = nn.Sequential(nn.Linear(in_features=self.in_features, out_features=fc_hidden_size), nn.ReLU(), nn.Linear(in_features=fc_hidden_size, out_features=out_features))

    def forward(self, perceptual_emb: torch.Tensor, language_emb: torch.Tensor) -> torch.Tensor:
        if len(perceptual_emb.shape) == 2:
            perceptual_emb = perceptual_emb[:, None, :] # add dimension to make seq_len = 1
            
        batch_size, seq_len, in_features = perceptual_emb.shape
        assert seq_len <= self.max_position_embeddings, f'Maximum supported ctx length is {self.max_position_embeddings}, you passed in {seq_len}'
        
        perceptual_emb = (
            torch.cat([perceptual_emb, torch.zeros((batch_size, seq_len, self.pad)).to(perceptual_emb.device)], dim=-1)
            if self.padding
            else perceptual_emb
        )
        position_ids = torch.arange(seq_len, dtype=torch.long, device=perceptual_emb.device).unsqueeze(0)
        position_embeddings = self.position_embeddings(position_ids)
        
        x = perceptual_emb + position_embeddings
        x = self.cross_attention(x, self.lang_projection(language_emb))
        x = self.fc(x)
        x = torch.mean(x, dim=1)  # gather all the sequence info

        return x

class TransformerEvaluationWrapper(torch.nn.Module):
    def __init__(self, model, device='cpu') -> None:
        super().__init__()
        self.model = model
        self.model.to(device)
        self.device = device
    
    def forward(self, lang, state):
        if len(state.shape) == 1:
            state = state[None, :]
        ret =  self.model(state.to(self.device), lang.to(self.device))
        return ret

if __name__ == '__main__':
    bsz = 10
    inp_sz = 32
    lang_embed =  torch.zeros((bsz, 69, 4096))
    t = ActionTransformer(inp_sz, 32, decoder_hidden_size=4096, num_layers=8)
    from torchinfo import  summary
    summary(t)
    out = t.forward(torch.zeros((bsz, inp_sz)), language_emb=lang_embed)
    print(out.shape)
    