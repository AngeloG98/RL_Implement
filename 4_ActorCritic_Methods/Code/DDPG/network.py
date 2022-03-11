from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, min_action) -> None:
        super(ActorNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        # projection
        self.a = (max_action - min_action)/2
        self.b = (max_action + min_action)/2

    def forward(self, x):
        y = self.net(x)
        out = self.a * y + self. b
        return out

class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super(CriticNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim+action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, u):
        out = self.net(torch.cat([x, u], 1))
        return out