import numpy as np
from functional import seq
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Net(nn.Module):
    def __init__(self, input_size, layer_dims, output_size, last_activation):
        super(Net, self).__init__()

        self.last_activation = last_activation

        layerdims = [input_size] + layer_dims + [output_size]
        self.layers = nn.ModuleList(seq(layerdims[:-1])
                                    .zip(layerdims[1:])
                                    .map(lambda dims: nn.Linear(dims[0], dims[1]))
                                    .to_list())

    def forward(self, inp):
        out = inp
        for layer in self.layers[:-1]:
            out = F.relu(layer(out))
        if self.last_activation:
            out = self.last_activation(self.layers[-1](out))
        else:
            out = self.layers[-1](out)
        return out
