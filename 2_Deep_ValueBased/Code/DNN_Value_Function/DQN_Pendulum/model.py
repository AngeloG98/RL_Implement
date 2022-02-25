from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# fc
class DQN_fnn(nn.Module):
    def __init__(self, layer_sizes):
        super(DQN_fnn, self).__init__()
        assert len(layer_sizes) > 1
        layers = []
        for index in range(len(layer_sizes) - 1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index + 1])
            act = nn.ReLU() if index < len(layer_sizes) - 2 else nn.Identity()
            layers += (linear, act)

        self.layer_sizes = layer_sizes
        self.fc = nn.Sequential(*layers)
    
    def forward(self, x):
        y = self.fc(x)
        return y

class dueling_DQN_fnn(nn.Module):
    def __init__(self, layer_sizes):
        super(dueling_DQN_fnn, self).__init__()
        assert len(layer_sizes) > 1
        self.layer_sizes = layer_sizes

        layers = []
        for index in range(len(layer_sizes) - 2):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index + 1])
            act = nn.ReLU()
            layers += (linear, act)

        self.fc1 = nn.Sequential(*layers)
        self.fc2 = nn.Sequential(*layers)
        self.value_layer = nn.Linear(layer_sizes[-2], 1)
        self.adv_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])

    def forward(self, x):
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        value = self.value_layer(y1)
        adv = self.adv_layer(y2)
        adv_average = torch.mean(adv, dim=0, keepdim=True)
        return value + adv - adv_average