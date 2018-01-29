import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, nb_actions=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)

        self.conv_2_hidden = nn.Linear(9248, 128)

        self.value_net = nn.Linear(128, 1)
        self.policy_net = nn.Linear(128, nb_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.conv_2_hidden(x))

        value = self.value_net(x)
        policy = F.softmax(self.policy_net(x), 1)
        return policy, value
