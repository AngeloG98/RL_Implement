import torch
import torch.nn as nn
import torch.nn.functional as F

class PG_fc_gaussian(nn.Module):
    def __init__(self, layer_sizes) -> None:
        super(PG_fc_gaussian, self).__init__()
        assert len(layer_sizes) > 1, 'Neural networks contains at least two layers！'
        
        mean_layers = []
        for index in range(len(layer_sizes)-1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index+1])
            act = nn.ReLU() if index < len(layer_sizes)-2 else nn.Tanh()
            mean_layers += (linear, act)
        
        std_layers = []
        for index in range(len(layer_sizes)-1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index+1])
            act = nn.ReLU() if index < len(layer_sizes)-2 else nn.Identity() #nn.Softplus()
            std_layers += (linear, act)

        self.mean = nn.Sequential(*mean_layers)
        self.std = nn.Sequential(*std_layers)
        self.layer_sizes = layer_sizes

    def forward(self, x):
        mu = self.mean(x) * 2
        # sigma = self.std(x)
        log_std = self.std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        sigma = log_std.exp()
        return mu, sigma

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
