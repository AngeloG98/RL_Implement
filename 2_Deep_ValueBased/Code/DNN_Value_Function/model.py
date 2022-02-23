from turtle import forward
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

# one hidden layer fnn
class DQN_fnn(nn.Module):
    def __init__(self, obs_shape, num_actions, hidden_layer_size = 50):
        super(DQN_fnn, self).__init__()
        self._obs_shape = obs_shape[0]
        self._num_actions = num_actions

        self.fc = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, num_actions)
        )
    
    def forward(self, x):
        return self.fc(x)

class Net(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(obs_shape[0], 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, num_actions)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value