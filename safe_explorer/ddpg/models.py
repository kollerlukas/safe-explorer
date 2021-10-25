import torch
from safe_explorer.core.net import Net


class Critic(Net):
    def __init__(self, input_size, layer_dims, output_size):
        super(Critic, self).__init__(input_size, layer_dims, output_size, None)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return super().forward(x)


class Actor(Net):
    def __init__(self, input_size, layer_dims, output_size):
        super(Actor, self).__init__(input_size,
                                    layer_dims, output_size, torch.tanh)
