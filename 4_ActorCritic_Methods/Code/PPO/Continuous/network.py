import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super(ActorCriticNet, self).__init__()

        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.std_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.critic_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        mu = 2*self.mean_net(x)

        std_raw = self.std_net(x)
        std_raw = torch.clamp(std_raw, min=-4.5) # Importante !
        sigma = F.softplus(std_raw)

        value = self.critic_net(x)
        return mu, sigma, value

