from functional import seq
import numpy as np
import torch

from safe_explorer.core.config import Config
from safe_explorer.core.tensorboard import TensorBoard
from safe_explorer.env.ballnd import BallND
from safe_explorer.env.spaceship import Spaceship
# from safe_explorer.ddpg.actor import Actor
# from safe_explorer.ddpg.critic import Critic
# from safe_explorer.ddpg.ddpg import DDPG
from safe_explorer.safety_layer.safety_layer import SafetyLayer

from safe_explorer.ddpg.ddpg import DDPGagent
from safe_explorer.ddpg.utils import OUNoise

import gym

class Trainer:
    def __init__(self):
        self._config = Config.get().main.trainer
        self._set_seeds()

    def _set_seeds(self):
        torch.manual_seed(self._config.seed)
        np.random.seed(int(self._config.seed))

    def _print_ascii_art(self):
        print(
        """
          _________       _____        ___________              .__                              
         /   _____/____ _/ ____\____   \_   _____/__  _________ |  |   ___________   ___________ 
         \_____  \\__  \\   __\/ __ \   |    __)_\  \/  /\____ \|  |  /  _ \_  __ \_/ __ \_  __ \\
         /        \/ __ \|  | \  ___/   |        \>    < |  |_> >  |_(  <_> )  | \/\  ___/|  | \/
        /_______  (____  /__|  \___  > /_______  /__/\_ \|   __/|____/\____/|__|    \___  >__|   
                \/     \/          \/          \/      \/|__|                           \/    
        """)                                                                                                                  

    def train(self):
        self._print_ascii_art()
        print("============================================================")
        print("Initialized SafeExplorer with config:")
        print("------------------------------------------------------------")
        Config.get().pprint()
        print("============================================================")

        env = BallND() if self._config.task == "ballnd" else Spaceship()

        # env = gym.make('MountainCarContinuous-v0')

        safety_layer = None
        if self._config.use_safety_layer:
            safety_layer = SafetyLayer(env)
            safety_layer.train()
        
        # state_dim = (seq(env.observation_space.spaces.values())
        #                     .map(lambda x: x.shape[0])
        #                     .sum())
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # actor = Actor(observation_dim, action_dim)
        # critic = Critic(observation_dim, action_dim)

        # safe_action_func = safety_layer.get_safe_action if safety_layer else None
        # ddpg = DDPG(env, actor, critic, safe_action_func)

        # ddpg.train()

        # get config
        config = Config.get().ddpg.trainer
        epochs = config.epochs
        training_episodes, evaluation_episodes = config.training_episodes, config.evaluation_episodes
        max_episode_length = config.max_episode_length
        batch_size = config.batch_size

        # create agent
        agent = DDPGagent(state_dim, action_dim)
        noise = OUNoise(env.action_space)

        # metrics
        cum_constr_viol = 0 # cumulative constraint violations
        disc_return = 0 # discounted return
        global_eval_step = 0

        # Tensorboard writer
        writer = TensorBoard.get_writer()

        for epoch in range(epochs):
            # training phase
            for episode in range(training_episodes):
                state = env.reset()
                noise.reset()
                
                constraints = env.get_constraint_values()

                for step in range(max_episode_length):
                    # get original policy action
                    action = agent.get_action(state)
                    # add noise
                    action = noise.get_action(action, step)
                    # get safe action
                    if safety_layer:
                        action = safety_layer.get_safe_action(state, action, constraints)
                    # apply action
                    new_state, reward, done, _ = env.step(action)
                    # push to memory
                    agent.memory.push(state, action, reward, new_state, done)
                    if len(agent.memory) > batch_size:
                        agent.update(batch_size)
                    # check if episode is done
                    if done:
                        break
                    else: 
                        state = new_state
                        constraints = env.get_constraint_values()
            print(f"Finished epoch {epoch}. Running validation ...")
            # evaluation phase
            episode_rewards, episode_lengths, episode_actions = [], [], []
            for episode in range(evaluation_episodes):
                state = env.reset()
                noise.reset()
                
                episode_action, episode_reward, episode_length, disc_return = 0, 0, 0, 0
                constraints = env.get_constraint_values()

                for step in range(max_episode_length):
                    # env.render()
                    # get original policy action
                    action = agent.get_action(state)
                    # add noise
                    action = noise.get_action(action, step)
                    # get safe action
                    if safety_layer:
                        action = safety_layer.get_safe_action(state, action, constraints)
                    episode_action += np.absolute(action)
                    # apply action
                    state, reward, done, info = env.step(action)
                    episode_reward += reward
                    disc_return += (config.discount_factor ** episode_length) * reward
                    constraints = env.get_constraint_values()
                    episode_length += 1
                    # check if episode is done
                    if done or step == max_episode_length - 1:
                        episode_rewards.append(episode_reward)
                        episode_lengths.append(episode_length)
                        episode_actions.append(episode_action / episode_length)
                        if 'constraint_violation' in info and info['constraint_violation']:
                            cum_constr_viol += 1
                        writer.add_scalar("episode reward", episode_reward, global_eval_step)
                        writer.add_scalar("discounted return", disc_return, global_eval_step)
                        writer.add_scalar("cumulative constraint violations", cum_constr_viol, global_eval_step)
                        global_eval_step += 1
                        break

            mean_episode_reward = np.mean(episode_rewards)
            mean_episode_length = np.mean(episode_lengths)

            print("Validation completed:\n"
              f"Number of episodes: {len(episode_actions)}\n"
              f"Average episode length: {mean_episode_length}\n"
              f"Average reward: {mean_episode_reward}\n"
              f"Average action magnitude: {np.mean(episode_actions)}\n"
              f"Discounted Return: {disc_return}\n"
              f"Cumulative Constraint Violations: {cum_constr_viol}")

        writer.close()

if __name__ == '__main__':
    Trainer().train()