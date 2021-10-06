from datetime import datetime
from functional import seq
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam

from safe_explorer.core.config import Config
from safe_explorer.core.replay_buffer import ReplayBuffer
from safe_explorer.core.tensorboard import TensorBoard
from safe_explorer.safety_layer.constraint_model import ConstraintModel

class SafetyLayer:
    def __init__(self, env):
        self._env = env

        self._config = Config.get().safety_layer.trainer

        self._num_constraints = env.get_num_constraints()

        # init constraint model
        state_dim = env.observation_space.shape[0] # ["agent_position"]
        action_dim = env.action_space.shape[0]
        self._models = [ConstraintModel(state_dim, action_dim) \
                        for _ in range(self._num_constraints)]
        self._optimizers = [Adam(x.parameters(), lr=self._config.lr) for x in self._models]

        self._replay_buffer = ReplayBuffer(self._config.replay_buffer_size)

        # Tensorboard writer
        self._writer = TensorBoard.get_writer()
        self._train_global_step = 0
        self._eval_global_step = 0

    def _eval_mode(self):
        for x in self._models:
            x.eval()

    def _train_mode(self):
        for x in self._models:
            x.train()

    def _sample_steps(self, num_steps):
        episode_length = 0

        observation = self._env.reset()

        for step in range(num_steps):            
            action = self._env.action_space.sample()
            c = self._env.get_constraint_values()
            observation_next, _, done, _ = self._env.step(action)
            c_next = self._env.get_constraint_values()

            self._replay_buffer.add({
                "action": action,
                "observation": observation, # ["agent_position"],
                "c": c,
                "c_next": c_next 
            })
            
            observation = observation_next            
            episode_length += 1
            
            if done or (episode_length == self._config.max_episode_length):
                observation = self._env.reset()
                episode_length = 0

    def _evaluate_batch(self, batch):
        observations = torch.Tensor(batch["observation"])
        actions = torch.Tensor(batch["action"])
        c = torch.Tensor(batch["c"])
        c_next = torch.Tensor(batch["c_next"])
        
        gs = [x(observations) for x in self._models]

        c_next_predicted = [c[:, i] + \
                            torch.bmm(x.view(x.shape[0], 1, -1), actions.view(actions.shape[0], -1, 1)).view(-1) \
                            for i, x in enumerate(gs)]
        losses = [torch.mean((c_next[:, i] - c_next_predicted[i]) ** 2) for i in range(self._num_constraints)]
        
        return losses

    def _update_batch(self, batch):
        batch = self._replay_buffer.sample(self._config.batch_size)

        # Update critic
        for x in self._optimizers:
            x.zero_grad()
        losses = self._evaluate_batch(batch)
        for x in losses:
            x.backward()
        for x in self._optimizers:
            x.step()

        return np.asarray([x.item() for x in losses])

    def evaluate(self):
        # Sample steps
        self._sample_steps(self._config.evaluation_steps)

        self._eval_mode()
        # compute losses
        losses = [list(map(lambda x: x.item(), self._evaluate_batch(batch))) for batch in \
                self._replay_buffer.get_sequential(self._config.batch_size)]

        losses = np.mean(np.concatenate(losses).reshape(-1, self._num_constraints), axis=0)

        self._replay_buffer.clear()
        # Log to tensorboard
        for x in enumerate(losses):
            self._writer.add_scalar(f"constraint {x[0]}/eval loss", x[1], self._eval_global_step)
        self._eval_global_step += 1

        self._train_mode()

        print(f"Validation completed, average loss {losses}")

    def get_safe_action(self, observation, action, c):    
        # Find the values of G
        self._eval_mode()
        g = [x(torch.Tensor(observation).view(1, -1)) for x in self._models] # ["agent_position"]
        self._train_mode()

        # Find the lagrange multipliers
        g = [x.data.detach().numpy().reshape(-1) for x in g]
        multipliers = [(np.dot(g_i, action) + c_i) / np.dot(g_i, g_i) for g_i, c_i in zip(g, c)]
        multipliers = [np.clip(x, 0, np.inf) for x in multipliers]

        # Calculate correction
        correction = np.max(multipliers) * g[np.argmax(multipliers)]

        action_new = action - correction
        return action_new

    def train(self):
        start_time = time.time()

        print("==========================================================")
        print("Initializing constraint model training...")
        print("----------------------------------------------------------")        
        print(f"Start time: {datetime.fromtimestamp(start_time)}")
        print("==========================================================")

        for epoch in range(self._config.epochs):
            # Just sample episodes for the whole epoch
            self._sample_steps(self._config.steps_per_epoch)
            
            # Do the update from memory
            losses = np.mean(np.concatenate([self._update_batch(batch) for batch in \
                    self._replay_buffer.get_sequential(self._config.batch_size)]).reshape(-1, self._num_constraints), axis=0)

            self._replay_buffer.clear()

            # Write losses and histograms to tensorboard
            for x in enumerate(losses):
                self._writer.add_scalar(f"constraint {x[0]}/training loss", x[1], self._train_global_step)
            self._train_global_step += 1

            print(f"Finished epoch {epoch} with losses: {losses}. Running validation ...")
            self.evaluate()
            print("----------------------------------------------------------")
        
        self._writer.close()
        print("==========================================================")
        print(f"Finished training constraint model. Time spent: {(time.time() - start_time) // 1} secs")
        print("==========================================================")