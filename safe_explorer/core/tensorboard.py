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
            taskname = ''
            safety_layer, reward_shaping = config.main.trainer.use_safety_layer, False
            if config.main.trainer.task == "ballnd":
                taskname = f'{config.main.trainer.task}-{config.env.ballnd.n}D_DDPG'
                if config.env.ballnd.control_acceleration:
                    taskname += '-acceleration'
                reward_shaping = config.env.ballnd.enable_reward_shaping
            elif config.main.trainer.task == 'spaceship':
                if config.env.spaceship.arena:
                    taskname = f'{config.main.trainer.task}-arena_DDPG'
                else:
                    taskname = f'{config.main.trainer.task}-corridor_DDPG'
                reward_shaping = config.env.spaceship.enable_reward_shaping
            outdir = f'runs/{taskname}' \
                + (f'+safety_layer' if safety_layer else '') \
                + (f'+reward_shaping' if reward_shaping else '')
            cls._writer = SummaryWriter(
                outdir + f'-({datetime.now().strftime("%b%d_%H-%M-%S")})')
            return cls._writer
