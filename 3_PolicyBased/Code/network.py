import torch
import torch.nn as nn
import torch.nn.functional as F

class PG_fc(nn.Module):
    def __init__(self, layer_sizes) -> None:
        super(PG_fc, self).__init__()
        assert len(layer_sizes) > 1, 'Neural networks contains at least two layers！'
        
        layers = []
        for index in range(len(layer_sizes)-1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index+1])
            act = nn.ReLU() if index < len(layer_sizes)-2 else nn.Identity()
            layers += (linear, act)

        self.fc = nn.Sequential(*layers)
        self.layer_sizes = layer_sizes

    def forward(self, x):
        y = self.fc(x)
        out = F.softmax(y, dim=1)
        return out

class PG_value(nn.Module):
    def __init__(self, layer_sizes) -> None:
        super(PG_value, self).__init__()
        assert len(layer_sizes) > 1, 'Neural networks contains at least two layers！'
        assert layer_sizes[-1] == 1, 'Dimension of out layers should be one！'
        
        layers = []
        for index in range(len(layer_sizes)-1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index+1])
            act = nn.ReLU() if index < len(layer_sizes)-2 else nn.Identity()
            layers += (linear, act)

        self.fc = nn.Sequential(*layers)
        self.layer_sizes = layer_sizes
    
    def forward(self, x):
        y = self.fc(x)
        return y
