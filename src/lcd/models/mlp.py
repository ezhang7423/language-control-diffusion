from torch import nn


class ForwardModel(nn.Module):
    def __init__(
        self, in_dim=(39 + 40 + 40 + 1), diffusion_dim=16, final_dim=39, num_units=[512, 512, 512]
    ):
        super(ForwardModel, self).__init__()
        actor_layers = []
        for out_dim in num_units:
            actor_layers.append(nn.Linear(in_dim, out_dim))
            actor_layers.append(nn.ReLU())
            in_dim = out_dim

        # Final layers
        actor_layers += [nn.Linear(in_dim, diffusion_dim), nn.ReLU(), nn.Linear(diffusion_dim, final_dim)]
        self.actor_layers = nn.Sequential(*actor_layers)
        self.depth = len(num_units)

    def forward(self, inputs, dropout_rate=0.0):
        return self.actor_layers(inputs)
