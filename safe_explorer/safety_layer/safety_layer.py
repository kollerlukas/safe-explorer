from datetime import datetime
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam

from safe_explorer.core.config import Config
from safe_explorer.core.tensorboard import TensorBoard
from safe_explorer.safety_layer.constraint_model import ConstraintModel
from safe_explorer.safety_layer.utils import Memory


class SafetyLayer:
    def __init__(self, env):
        self.env = env
        # get config
        config = Config.get().safety_layer.trainer
        # set attributes
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.epochs = config.epochs
        self.training_steps_per_epoch = config.training_steps_per_epoch
        self.evaluation_steps_per_epoch = config.evaluation_steps_per_epoch
        self.memory_buffer_size = config.memory_buffer_size
        self.sample_data_episodes = config.sample_data_episodes
        # init constraint model
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.models = [ConstraintModel(state_dim, action_dim)
                       for _ in range(env.get_num_constraints())]
        self.optims = [Adam(model.parameters(), lr=self.lr)
                       for model in self.models]
        # use doubles for calculations
        for model in self.models:
            model.double()
        # Mean-Squared-Error as loss criterion
        self.loss_criterion = nn.MSELoss()
        # init memory
        self.memory = Memory(self.memory_buffer_size)
        # Tensorboard writer
        self.writer = TensorBoard.get_writer()
        self.train_step = 0
        self.eval_step = 0

    def _sample_steps(self, episodes):
        # sample episodes and push to memory
        for _ in range(episodes):
            state = self.env.reset(random_agent_position=True)
            constraints = self.env.get_constraint_values()

            done = False
            while not done:
                # get random action
                action = self.env.action_space.sample()
                # apply action
                next_state, _, done, _ = self.env.step(action)
                # get changed constraint values
                next_constraints = self.env.get_constraint_values()
                # push to memory
                self.memory.push(state, action, constraints, next_constraints)
                state = next_state
                constraints = next_constraints

    def _calc_loss(self, model, states, actions, constraints, next_constraints):
        # calculate batch-dot-product via torch.einsum
        gi = model.forward(states)
        predicted_constraints = constraints + \
            torch.einsum('ij,ij->i', gi, actions)
        # alternative: calculate batch-dot-product via torch.bmm
        # gi = model.forward(states).unsqueeze(1)
        # actions = actions.unsqueeze(1)
        # predicted_constraints = constraints + torch.bmm(gi, actions.transpose(1,2)).squeeze(1).squeeze(1)

        # alternative loss calculation
        # loss = (next_constraints - predicted_constraints) ** 2
        # return torch.mean(loss)
        loss = self.loss_criterion(next_constraints, predicted_constraints)
        return loss

    def _update_batch(self, batch):
        states, actions, constraints, next_constraints = batch
        states = torch.DoubleTensor(states)
        actions = torch.DoubleTensor(actions)
        constraints = torch.DoubleTensor(constraints)
        next_constraints = torch.DoubleTensor(next_constraints)

        losses = []
        for i, (model, optimizer) in enumerate(zip(self.models, self.optims)):
            # calculate loss
            loss = self._calc_loss(
                model, states, actions, constraints[:, i], next_constraints[:, i])
            losses.append(loss.item())
            # zero gradients
            optimizer.zero_grad()
            # backward pass
            loss.backward()
            # update optimizer
            optimizer.step()

        return np.array(losses)

    def _evaluate_batch(self, batch):
        states, actions, constraints, next_constraints = batch
        states = torch.DoubleTensor(states)
        actions = torch.DoubleTensor(actions)
        constraints = torch.DoubleTensor(constraints)
        next_constraints = torch.DoubleTensor(next_constraints)

        losses = []
        for i, model in enumerate(self.models):
            # compute losses
            loss = self._calc_loss(
                model, states, actions, constraints[:, i], next_constraints[:, i])
            losses.append(loss.item())

        return np.array(losses)

    def train(self):
        start_time = time.time()

        print("==========================================================")
        print("Initializing constraint model training...")
        print("----------------------------------------------------------")
        print(f"Start time: {datetime.fromtimestamp(start_time)}")
        print("==========================================================")

        # sample random action episodes and store them in memory
        self._sample_steps(self.sample_data_episodes)

        for epoch in range(self.epochs):
            # training phase
            losses = []
            for _ in range(self.training_steps_per_epoch):
                batch = self.memory.sample(self.batch_size)
                loss = self._update_batch(batch)
                losses.append(loss)
            print(
                f"Finished epoch {epoch} with average loss: {np.mean(losses, axis=0)}. Running evaluation ...")
            # log training losses to tensorboard
            for i, loss in enumerate(np.mean(losses, axis=0)):
                self.writer.add_scalar(
                    f"constraints/constraint {i}/training loss", loss, self.train_step)
            self.train_step += 1

            # evaluation phase
            losses = []
            for _ in range(self.evaluation_steps_per_epoch):
                batch = self.memory.sample(self.batch_size)
                loss = self._evaluate_batch(batch)
                losses.append(loss)
            print(
                f"Evaluation completed, average loss {np.mean(losses, axis=0)}")
            # log evaluation losses to tensorboard
            for i, loss in enumerate(np.mean(losses, axis=0)):
                self.writer.add_scalar(
                    f"constraints/constraint {i}/eval loss", loss, self.eval_step)
            self.eval_step += 1
            print("----------------------------------------------------------")

        self.writer.close()
        print("==========================================================")
        print(
            f"Finished training constraint model. Time spent: {(time.time() - start_time) // 1} secs")
        print("==========================================================")

    def get_safe_action(self, state, action, constraints):
        state = torch.DoubleTensor(state)
        action = torch.DoubleTensor(action)
        constraints = torch.DoubleTensor(constraints)

        g = [model.forward(state) for model in self.models]
        # calculate lagrange multipliers
        multipliers = [torch.clip((torch.dot(
            gi, action) + ci) / torch.dot(gi, gi), min=0) for gi, ci in zip(g, constraints)]
        # Calculate correction
        safe_action = action - np.max(multipliers) * g[np.argmax(multipliers)]

        return safe_action.data.detach().numpy()

    # def predict_constraints(self, state, action, constraints):
    #     state = torch.DoubleTensor(state)
    #     action = torch.DoubleTensor(action)
    #     constraints = torch.DoubleTensor(constraints)

    #     g = [model.forward(state) for model in self.models]
    #     # calculate lagrange multipliers
    #     pred_constraints = [(torch.dot(gi, action) + ci) for gi, ci in zip(g, constraints)]
    #     pred_constraints = [ci.data.detach().numpy() for ci in pred_constraints]
    #     return pred_constraints
