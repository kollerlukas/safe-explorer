from torch.nn.init import uniform_

from safe_explorer.core.config import Config
from safe_explorer.core.net import Net

class ConstraintModel(Net):
    def __init__(self, input_size, output_size):
        config = Config.get().safety_layer.constraint_model
        super(ConstraintModel, self).__init__(input_size, config.layers, output_size, None)