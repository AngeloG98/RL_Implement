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

        layers_adv = []
        for index in range(len(layer_sizes) - 1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index + 1])
            act = nn.ReLU() if index < len(layer_sizes) - 2 else nn.Identity()
            layers_adv += (linear, act)
        
        layers_value = []
        for index in range(len(layer_sizes) - 1):
            if index == len(layer_sizes) - 2:
                linear = nn.Linear(layer_sizes[index], 1)
                act = nn.Identity()
                layers_value += (linear, act)
            else:
                linear = nn.Linear(layer_sizes[index], layer_sizes[index + 1])
                act = nn.ReLU()
                layers_value += (linear, act)

        self.fc_adv = nn.Sequential(*layers_adv)
        self.fc_value = nn.Sequential(*layers_value)

    def forward(self, x):
        value = self.fc_value(x)
        adv = self.fc_adv(x)
        adv_average = torch.mean(adv, dim=0, keepdim=True)
        return value + adv - adv_average