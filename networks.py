import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import gym
import numpy as np
import warnings; warnings.filterwarnings('ignore')
from matplotlib.animation import FuncAnimation

DEVICE = torch.device("cuda")

# Define actor and critic networks
class actor_network(nn.Module):
    def __init__(self, n_actions, input_dims):
        super().__init__()

        self.apply(self.init_weights)
        self.fc1 = nn.Linear(input_dims, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, n_actions)
        self.log_std = nn.Linear(256, n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        self.device = torch.device('cuda')
        self.to(self.device)
    
    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    def forward(self, x):
        x = torch.tensor([x]).float().to(self.device) if type(x) is not torch.Tensor else x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        return dist

class critic_network(nn.Module):
    def __init__(self, input_dims):
        super().__init__()

        self.apply(self.init_weights)
        self.fc1 = nn.Linear(input_dims, 256)
        self.fc2 = nn.Linear(256, 1)

        self.optimizer = Adam(self.parameters(), lr=3e-4)
        self.to(DEVICE)
    
    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    def forward(self, x):
        x = torch.tensor([x]).float().to(DEVICE) if type(x) is not torch.Tensor else x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x