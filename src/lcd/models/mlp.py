import einops
import torch
from torch import nn


class ForwardModel(nn.Module):
    def __init__(
        self,
        in_dim=(39 + 40 + 40 + 1),
        diffusion_dim=16,
        final_dim=39,
        num_units=[512, 512, 512],
    ):
        super(ForwardModel, self).__init__()
        encoder = []
        for out_dim in num_units:
            encoder.append(nn.Linear(in_dim, out_dim))
            encoder.append(nn.ReLU())
            encoder.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        self.encoder = nn.Sequential(*encoder)

        final_layers = nn.Sequential(
            nn.Linear(diffusion_dim * 2, num_units[0]),
            nn.ReLU(),
            nn.LayerNorm(num_units[0]),
            nn.Linear(num_units[0], num_units[0]),
            nn.ReLU(),
            nn.LayerNorm(num_units[0]),
            nn.Linear(num_units[0], final_dim),
        )

        # Final layers
        self.final_layers = final_layers

        self.depth = len(num_units)

    def forward(self, obs, next_obs):
        # Concatenate obs and next_obs along the feature dimension
        combined_input = torch.cat([obs, next_obs], dim=0)

        # Pass the combined input through encoder
        encoded_combined = self.encoder(combined_input)

        # Concatenate the separately encoded obs and next_obs and pass through final layers
        return self.final_layers(
            einops.rearrange(encoded_combined, "(t n)  f -> n (t f)", t=2)
        )

    def encode(self, inputs):
        return self.encoder(inputs)
