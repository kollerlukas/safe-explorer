import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from functional import seq

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

class Critic(Net):
    def __init__(self, input_size, layer_dims, output_size, last_activation=None):
        super(Critic, self).__init__(input_size, layer_dims, output_size, last_activation)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        return super().forward(x)

class Actor(Net):
    def __init__(self, input_size, layer_dims, output_size, last_activation=torch.tanh):
        super(Actor, self).__init__(input_size, layer_dims, output_size, last_activation)

# class Critic(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(Critic, self).__init__()
#         self.linear1 = nn.Linear(input_size, 500)
#         self.linear2 = nn.Linear(500, 500)
#         self.linear3 = nn.Linear(500, 500)
#         self.linear4 = nn.Linear(500, output_size)

#     def forward(self, state, action):
#         """
#         Params state and actions are torch tensors
#         """
#         x = torch.cat([state, action], 1)
#         x = F.relu(self.linear1(x))
#         x = F.relu(self.linear2(x))
#         x = F.relu(self.linear3(x))
#         x = self.linear4(x)

#         return x

# class Actor(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, learning_rate = 3e-4):
#         super(Actor, self).__init__()
#         self.linear1 = nn.Linear(input_size, 100)
#         self.linear2 = nn.Linear(100, 100)
#         self.linear3 = nn.Linear(100, 100)
#         self.linear4 = nn.Linear(100, output_size)      
    
#     def forward(self, state):
#         """
#         Param state is a torch tensor
#         """
#         x = F.relu(self.linear1(state))
#         x = F.relu(self.linear2(x))
#         x = F.relu(self.linear3(x))
#         x = torch.tanh(self.linear4(x))
#         return x

