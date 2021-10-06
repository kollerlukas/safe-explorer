#
# Based on implementation from: https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
#

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable

from safe_explorer.core.config import Config
from safe_explorer.ddpg.models import Critic, Actor
from safe_explorer.ddpg.utils import Memory
from safe_explorer.core.tensorboard import TensorBoard

class DDPGagent:
    def __init__(self, state_dim, action_dim):
        config = Config.get().ddpg
        # Params
        self.gamma = config.trainer.discount_factor # gamma
        self.tau = config.trainer.tau # tau

        # Networks
        self.actor = Actor(state_dim, config.actor.layers, action_dim)
        self.actor_target = Actor(state_dim, config.actor.layers, action_dim)
        self.critic = Critic(state_dim + action_dim, config.critic.layers, action_dim)
        self.critic_target = Critic(state_dim + action_dim, config.critic.layers, action_dim)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Training
        self.memory = Memory(config.trainer.replay_buffer_size)    
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = Adam(self.actor.parameters(), lr=config.trainer.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.trainer.critic_lr)

        # Tensorboard writer
        self._writer = TensorBoard.get_writer()
        self._train_step = 0
    
    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().numpy()[0,0]
        return action
    
    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
    
        # Critic loss        
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        # self._writer.add_scalar("DDPG/critic loss", critic_loss.item(), self._train_step)
        # self._writer.add_scalar("DDPG/actor loss", policy_loss.item(), self._train_step)
        # self._train_step +=1