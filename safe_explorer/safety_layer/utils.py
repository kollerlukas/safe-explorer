from collections import deque
import random

class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, constraints, next_constraints):
        experience = (state, action, constraints, next_constraints)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch, action_batch, constraints_batch, next_constraints_batch = [], [], [], []

        batch = random.sample(self.buffer, batch_size)

        for state, action, constraints, next_constraints in batch:
            state_batch.append(state)
            action_batch.append(action)
            constraints_batch.append(constraints)
            next_constraints_batch.append(next_constraints)
        
        return state_batch, action_batch, constraints_batch, next_constraints_batch

    def __len__(self):
        return len(self.buffer)

    def sample_batch_by_index(self, indices):
        state_batch, action_batch, constraints_batch, next_constraints_batch = [], [], [], []
        for i in indices:
            state, action, constraints, next_constraints = self.buffer[i]
            state_batch.append(state)
            action_batch.append(action)
            constraints_batch.append(constraints)
            next_constraints_batch.append(next_constraints)
        return state_batch, action_batch, constraints_batch, next_constraints_batch

    # def clear(self):
    #     self.buffer.clear()