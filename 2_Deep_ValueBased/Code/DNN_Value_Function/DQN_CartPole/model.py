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
        return self.fc(x)