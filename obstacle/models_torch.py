import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import namedtuple
import random


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    """ReplayMemory that holds transitions observed recently."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class SimpleFCDQN(nn.Module):
    """Simple DQN that maps the 6 sensors from the crazyflie to 5 actions for
    either staying still or evading in a direction."""
    def __init__(self, inputs, fcsize, actions):
        super(SimpleFCDQN, self).__init__()
        self.fc1 = nn.Linear(inputs, fcsize)
        self.fc2 = nn.Linear(fcsize, actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x 


