import torch
import torch.nn as nn
import numpy as np

class DQN_conv(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        feature_size = self._get_feature_size(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_feature_size(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        feature = self.conv(x).view(x.size()[0], -1)
        return self.fc(feature)


class DQN_conv_duel(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN_conv_duel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        feature_size = self._get_feature_size(input_shape)

        self.advantage = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_feature_size(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        feature = self.conv(x).view(x.size()[0], -1)
        adv = self.advantage(feature)
        val = self.value(feature)
        return val + adv - adv.mean()