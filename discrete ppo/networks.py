import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import gym
import numpy as np
import warnings; warnings.filterwarnings('ignore')
from matplotlib.animation import FuncAnimation

DEVICE = torch.device("cpu")

# Define actor and critic networks
class actor_network(nn.Module):
    def __init__(self, n_actions, input_dims):
        super().__init__()

        self.fc1 = nn.Linear(input_dims, 256)
        self.fc2 = nn.Linear(256, n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self, x):
        x = torch.tensor([x]).float().to(DEVICE) if type(x) is not torch.Tensor else x
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        dist = torch.distributions.Categorical(x)
        return dist


class critic_network(nn.Module):
    def __init__(self, input_dims):
        super().__init__()

        self.fc1 = nn.Linear(input_dims, 256)
        self.fc2 = nn.Linear(256, 1)

        self.optimizer = Adam(self.parameters(), lr=3e-4)
        self.to(DEVICE)

    def forward(self, x):
        x = torch.tensor([x]).float().to(DEVICE) if type(x) is not torch.Tensor else x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x