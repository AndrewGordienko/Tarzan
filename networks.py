import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from torch.optim import Adam

FC1_DIMS = 1024
FC2_DIMS = 512
DEVICE = torch.device("cpu")

# Define actor and critic networks
class actor_network(nn.Module):
    def __init__(self, n_actions, input_dims, alpha):
        super().__init__()

        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)

        self.optimizer = Adam(self.parameters(), lr=alpha)
        self.to(DEVICE)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

class critic_network(nn.Module):
    def __init__(self, input_dims, alpha):
        super().__init__()

        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 1)

        self.optimizer = Adam(self.parameters(), lr=alpha)
        self.to(DEVICE)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

        return x