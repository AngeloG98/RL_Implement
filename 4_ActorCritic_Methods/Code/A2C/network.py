import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticNet(nn.Module):
    def __init__(self, layer_sizes) -> None:
        super(ActorCriticNet, self).__init__()
        assert len(layer_sizes) > 1, 'Neural networks contains at least two layers！'
        
        actor_layers = []
        for index in range(len(layer_sizes)-1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index+1])
            act = nn.ReLU() if index < len(layer_sizes)-2 else nn.Identity()
            actor_layers += (linear, act)
        
        critic_layers = []
        for index in range(len(layer_sizes)-1):
            if index == len(layer_sizes)-2:
                linear = nn.Linear(layer_sizes[index], 1)
                act = nn.Identity()
                critic_layers += (linear, act)
            else:
                linear = nn.Linear(layer_sizes[index], layer_sizes[index + 1])
                act = nn.ReLU()
                critic_layers += (linear, act)

        self.actor_net = nn.Sequential(*actor_layers)
        self.critic_net = nn.Sequential(*critic_layers)
        self.layer_sizes = layer_sizes

    def forward(self, x):
        y = self.actor_net(x)
        policy = F.softmax(y, dim=1)
        value = self.critic_net(x)
        return policy, value

# class ActorCriticNet(nn.Module):
#     def __init__(self, layer_sizes) -> None:
#         super(ActorCriticNet, self).__init__()
#         assert len(layer_sizes) > 1, 'Neural networks contains at least two layers！'
        
#         layers = []
#         for index in range(len(layer_sizes)-2):
#             linear = nn.Linear(layer_sizes[index], layer_sizes[index+1])
#             act = nn.ReLU()
#             layers += (linear, act)
#         self.fc = nn.Sequential(*layers)

#         self.actor_net = nn.Sequential(
#             nn.Linear(layer_sizes[-2], 48),
#             nn.ReLU(),
#             nn.Linear(48, layer_sizes[-1])
#         )
#         self.critic_net = nn.Sequential(
#             nn.Linear(layer_sizes[-2], 48),
#             nn.ReLU(),
#             nn.Linear(48, 1)
#         )
#         self.layer_sizes = layer_sizes

#     def forward(self, x):
#         y = self.fc(x)
#         policy = F.softmax(self.actor_net(y), dim=1)
#         value = self.critic_net(y)
#         return policy, value
