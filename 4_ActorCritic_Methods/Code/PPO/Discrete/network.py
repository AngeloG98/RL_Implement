import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super(ActorCriticNet, self).__init__()

        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        policy = self.actor_net(x)
        value = self.critic_net(x)
        return policy, value

