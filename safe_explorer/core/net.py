from functional import seq
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_size, layer_dims, output_size, last_activation):
        super(Net, self).__init__()

        self._last_activation = last_activation

        layer_dims = [input_size] + layer_dims + [output_size]
        self._layers = nn.ModuleList(seq(layer_dims[:-1])
                                     .zip(layer_dims[1:])
                                     .map(lambda x: nn.Linear(x[0], x[1]))
                                     .to_list())

    def forward(self, inp):
        out = inp
        for layer in self._layers[:-1]:
            out = F.relu(layer(out))
        if self._last_activation:
            out = self._last_activation(self._layers[-1](out))
        else:
            out = self._layers[-1](out)
        return out
