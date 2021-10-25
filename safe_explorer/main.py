import time
from datetime import datetime
from functional import seq
import numpy as np
import torch

from safe_explorer.core.config import Config
from safe_explorer.core.tensorboard import TensorBoard
from safe_explorer.env.ballnd import BallND
from safe_explorer.safety_layer.safety_layer import SafetyLayer

from safe_explorer.ddpg.ddpg import DDPGAgent
from safe_explorer.ddpg.utils import OUNoise


class Trainer:
    def __init__(self):
        config = Config.get().main.trainer
        # set seeds
        # torch.manual_seed(config.seed)
        # np.random.seed(int(config.seed))

        self.use_safety_layer = config.use_safety_layer

    def train(self):
        print("============================================================")
        print("Initialized SafeExplorer with config:")
        print("------------------------------------------------------------")
        Config.get().pprint()
        print("============================================================")

        # create ball domain
        env = BallND()

        # init Safety Layer
        safety_layer = None
        if self.use_safety_layer:
            safety_layer = SafetyLayer(env)
            safety_layer.train()
        # obtain state and action dimensions
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # get config
        config = Config.get().ddpg.trainer
        # get relevant config values
        epochs = config.epochs
        # training_steps = config.training_steps_per_epoch
        training_episodes = config.training_episodes_per_epoch
        evaluation_episodes = config.evaluation_episodes_per_epoch
        # max_episode_length = config.max_episode_length
        batch_size = config.batch_size

        # create agent
        agent = DDPGAgent(state_dim, action_dim)
        # create exploration noise
        noise = OUNoise(env.action_space)
        # metrics for tensorboard
        cum_constr_viol = 0  # cumulative constraint violations
        eval_step = 0
        episode_action, episode_reward, episode_length = 0, 0, 0
        # create Tensorboard writer
        writer = TensorBoard.get_writer()

        start_time = time.time()
        print("==========================================================")
        print("Initializing DDPG training...")
        print("----------------------------------------------------------")
        print(f"Start time: {datetime.fromtimestamp(start_time)}")
        print("==========================================================")

        for epoch in range(epochs):
            # training phase
            # episode_step, done = 0, True
            # for step in range(training_steps):
            #     if done:
            #         # reset episode
            #         state = env.reset()
            #         noise.reset()
            #         episode_step = 0
            #         constraints = env.get_constraint_values()
            #     # get original policy action
            #     action = agent.get_action(state)
            #     # add OU-noise for exploration
            #     action = noise.get_action(action, episode_step)
            #     # get safe action
            #     if safety_layer:
            #         action = safety_layer.get_safe_action(
            #             state, action, constraints)
            #     # apply action
            #     next_state, reward, done, info = env.step(action)
            #     episode_step += 1
            #     # push to memory
            #     agent.memory.push(state, action, reward, next_state, done)
            #     # update agent
            #     if len(agent.memory) > batch_size:
            #         agent.update(batch_size)
            #     if not done:
            #         state = next_state
            #         constraints = env.get_constraint_values()
            # print(f"Finished epoch {epoch}. Running evaluation ...")

            # training phase
            for _ in range(training_episodes):
                noise.reset()
                state = env.reset()
                constraints = env.get_constraint_values()
                episode_step = 0

                done = False
                while not done:
                    # get original policy action
                    action = agent.get_action(state)
                    # add OU-noise
                    action = noise.get_action(action, episode_step)
                    # get safe action
                    if safety_layer:
                        action = safety_layer.get_safe_action(
                            state, action, constraints)
                    # apply action
                    next_state, reward, done, info = env.step(action)
                    episode_step += 1
                    # push to memory
                    agent.memory.push(state, action, reward, next_state, done)
                    # update agent
                    if len(agent.memory) > batch_size:
                        agent.update(batch_size)
                    state = next_state
                    constraints = env.get_constraint_values()
            print(f"Finished epoch {epoch}. Running evaluation ...")

            # evaluation phase
            episode_rewards, episode_lengths, episode_actions = [], [], []
            for _ in range(evaluation_episodes):
                state = env.reset()
                constraints = env.get_constraint_values()

                episode_action, episode_reward, episode_length = 0, 0, 0
                done = False
                while not done:
                    # render environment; only for ball-1D
                    env.render()
                    # get original policy action
                    action = agent.get_action(state)
                    # get safe action
                    if safety_layer:
                        action = safety_layer.get_safe_action(
                            state, action, constraints)
                    episode_action += np.absolute(action)
                    # apply action
                    state, reward, done, info = env.step(action)
                    # update metrics
                    episode_reward += reward
                    constraints = env.get_constraint_values()
                    episode_length += 1

                if 'constraint_violation' in info and info['constraint_violation']:
                    cum_constr_viol += 1
                # log metrics to tensorboard
                writer.add_scalar("metrics/episode length",
                                  episode_length, eval_step)
                writer.add_scalar("metrics/episode reward",
                                  episode_reward, eval_step)
                writer.add_scalar("metrics/cumulative constraint violations",
                                  cum_constr_viol, eval_step)
                eval_step += 1

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_actions.append(episode_action / episode_length)

            print("Evaluation completed:\n"
                  f"Number of episodes: {len(episode_actions)}\n"
                  f"Average episode length: {np.mean(episode_lengths)}\n"
                  f"Average reward: {np.mean(episode_rewards)}\n"
                  f"Average action magnitude: {np.mean(episode_actions)}\n"
                  f"Cumulative Constraint Violations: {cum_constr_viol}")
            print("----------------------------------------------------------")
        print("==========================================================")
        print(
            f"Finished DDPG training. Time spent: {(time.time() - start_time) // 1} secs")
        print("==========================================================")
        # close environment
        env.close()
        # close tensorboard writer
        writer.close()


if __name__ == '__main__':
    Trainer().train()
