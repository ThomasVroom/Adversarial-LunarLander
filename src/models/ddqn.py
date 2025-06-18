import random
import numpy as np
from typing import List
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn

class Agent(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Agent, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        return self.network(x)

    def get_action(self, x):
        x = self.forward(x)
        return torch.argmax(x)

@dataclass
class ReplayBufferSample:
    state: any
    action: any
    reward: float
    new_state: any
    was_terminal: bool

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity) # discards the oldest elements when full

    def add_replay_buffer_sample(self, element: ReplayBufferSample):
        self.buffer.append(element) # add new element, handling capacity

    def add_new_sample(self, state, action, reward, new_state, was_terminal):
        sample = ReplayBufferSample(state, action, reward, new_state, was_terminal)
        self.add_replay_buffer_sample(sample)

    def sample_buffer(self, batch_size: int) -> List[ReplayBufferSample]:
        samples = random.sample(self.buffer, batch_size)
        return samples

    # variation that will return pytorch tensors on a target device,
    def get_pytorch_training_samples(self, device, learning_batch_size):
        samples = self.sample_buffer(learning_batch_size)
        states = torch.tensor(np.array([s.state for s in samples], dtype=np.float32)).to(device)
        actions = torch.tensor([s.action for s in samples]).to(device)
        rewards = torch.tensor([s.reward for s in samples], dtype=torch.float32).to(device)
        new_states = torch.tensor(np.array([s.new_state for s in samples], dtype=np.float32)).to(device)
        was_terminals = torch.tensor([s.was_terminal for s in samples]).to(device)
        return states, actions, rewards, new_states, was_terminals

    def __len__(self):
        return len(self.buffer)
