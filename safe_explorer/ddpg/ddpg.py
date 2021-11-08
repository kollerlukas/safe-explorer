#
# Based on the implementation from: https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
#

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable

from safe_explorer.core.config import Config
from safe_explorer.ddpg.models import Critic, Actor
from safe_explorer.ddpg.utils import Memory
from safe_explorer.core.tensorboard import TensorBoard


class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        config = Config.get().ddpg.trainer
        # set attributes
        self.memory_buffer_size = config.memory_buffer_size
        self.gamma = config.gamma
        self.tau = config.tau
        self.reward_scale = config.reward_scale
        self.actor_lr = config.actor_lr
        self.critic_lr = config.critic_lr
        self.actor_layers = config.actor_layers
        self.critic_layers = config.critic_layers
        self.actor_weight_decay = config.actor_weight_decay
        self.critic_weight_decay = config.critic_weight_decay
        # init actor and critic networks
        self.actor = Actor(state_dim, self.actor_layers, action_dim)
        self.actor_target = Actor(state_dim, self.actor_layers, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(state_dim + action_dim,
                             self.critic_layers, action_dim)
        self.critic_target = Critic(
            state_dim + action_dim, self.critic_layers, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # use doubles for calculations
        self.actor.double()
        self.actor_target.double()
        self.critic.double()
        self.critic_target.double()
        # Training
        self.memory = Memory(self.memory_buffer_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr,
                                    weight_decay=self.actor_weight_decay)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr,
                                     weight_decay=self.critic_weight_decay)
        # Tensorboard writer
        self.writer = TensorBoard.get_writer()
        self.train_step = 0

    def set_train_mode(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def set_eval_mode(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).double().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().numpy()[0]
        return action

    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(
            batch_size)
        states = torch.DoubleTensor(states)
        actions = torch.DoubleTensor(actions)
        rewards = torch.DoubleTensor(rewards)
        next_states = torch.DoubleTensor(next_states)

        # Critic loss
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards * self.reward_scale + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qprime, Qvals)

        # Actor loss
        policy_loss = - \
            self.critic.forward(states, self.actor.forward(states)).mean()

        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1.0 - self.tau))

        # self.writer.add_scalar("DDPG/critic loss", critic_loss.item(), self.train_step)
        # self.writer.add_scalar("DDPG/actor loss", policy_loss.item(), self.train_step)
        # self.train_step +=1
