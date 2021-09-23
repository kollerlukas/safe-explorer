#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter


class TensorBoard:
    _writer = None


    @classmethod
    def get_writer(cls):
        if cls._writer:
            return cls._writer
        cls._writer = SummaryWriter()
        
        return cls._writer