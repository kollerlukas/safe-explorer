from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from safe_explorer.core.config import Config


class TensorBoard:
    _writer = None

    @classmethod
    def get_writer(cls):
        if cls._writer:
            return cls._writer
        else:
            config = Config.get()
            outdir = None
            print(f'datetime: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
            outdir = 'runs/'
            if config.main.trainer.task == "ballnd":
                outdir += f'{config.main.trainer.task}-{config.env.ballnd.n}D_DDPG'
            elif config.main.trainer.task == 'spaceship':
                outdir += f'{config.main.trainer.task}_DDPG'
            outdir += (f'+safety_layer' if config.main.trainer.use_safety_layer else '') \
                + (f'+reward_shaping' if config.env.ballnd.enable_reward_shaping else '')
            cls._writer = SummaryWriter(
                outdir + f'-({datetime.now().strftime("%b%d_%H-%M-%S")})')
            return cls._writer
