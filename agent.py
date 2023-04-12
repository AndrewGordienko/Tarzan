import torch
from torch.optim import Adam
import torch.nn.functional as F
import torch.distributions.transforms as transforms
from torch.nn.utils import clip_grad_norm_
from torch.distributions.categorical import Categorical
import numpy as np


from ppo_memory import ReplayBuffer
from networks import actor_network, critic_network

BATCH_SIZE = 32
DEVICE = torch.device("cpu")

class Agent:
    def __init__(self, n_actions, input_dims, alpha=0.0001):
        self.gamma = 0.99
        self.policy_clip = 0.2
        self.n_epochs = 50
        self.gae_lambda = 0.95
        self.entropy_weight = 0.01

        self.actor = actor_network(n_actions, input_dims, alpha)
        self.critic = critic_network(input_dims, alpha)
        self.memory = ReplayBuffer()

        self.optimizer = Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=alpha)


    def remember(self, state, action, reward, done, probs, vals):
        self.memory.add(state, action, reward, done, probs, vals)

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(DEVICE)

        # Pass the state through the actor network to get the probability distribution over the actions
        probs = self.actor(state)

        # Create a Categorical distribution from the logits to represent the policy
        dist = Categorical(probs=probs)

        # Sample an action from the distribution
        action = dist.sample()

        # Compute the log probability of the chosen action
        log_prob = dist.log_prob(action).unsqueeze(0)

        # Pass the state through the critic network to get the estimated value function
        value = self.critic(state)

        # Squeeze the action and value tensors to remove any unnecessary dimensions
        action = action.squeeze().item()
        value = value.squeeze().item()

        # Detach the log_prob tensor from the computation graph
        log_prob = log_prob.detach().cpu()

        return action, log_prob, value

    def compute_advantages(self, rewards, dones, values):
        # Estimate how much better or worse an action is compared to the average action taken by the policy
        # Initialize the advantages and the cumulative advantage
        advantages = []
        cumulative_advantage = 0

        # Add a zero value at the end of the values tensor
        values = torch.cat([values, torch.tensor([0], dtype=torch.float32).to(DEVICE)], dim=0)

        # Cast the dones tensor to a float
        dones = dones.float()

        # Compute the advantages for each experience in reverse order
        for i in range(len(rewards) - 1, -1, -1):
            # Compute the temporal difference error for the current experience
            delta = rewards[i] + self.gamma * (1 - dones[i]) * values[i+1] - values[i]

            # Update the cumulative advantage
            cumulative_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * cumulative_advantage

            # Add the cumulative advantage to the list of advantages
            advantages.append(cumulative_advantage)

        # Convert the list of advantages to a PyTorch tensor and normalize
        advantages = torch.tensor(list(reversed(advantages)), dtype=torch.float32).to(DEVICE)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    
    def compute_new_probs_and_values(self, states, actions):
        # Pass the states through the actor network to get the logits for the probability distribution over the actions
        logits = self.actor(states)

        # Create a Categorical distribution from the logits to represent the policy
        dist = Categorical(logits=logits)

        # Get the new policy probabilities by getting the probs attribute from the distribution
        new_probs = dist.probs

        # Select the probabilities corresponding to the taken actions
        new_probs = new_probs[range(len(actions)), actions]

        # Pass the states through the value function network to compute the new value function estimates
        new_values = self.critic(states)

        return new_probs, new_values


    def compute_clipped_objective_and_value_loss(self, old_probs, new_probs, old_values, new_values, advantages):
        # Compute the value function loss using the mean squared error loss
        value_loss = F.mse_loss(old_values, new_values)

        # Reshape new_probs to match the shape of old_probs
        new_probs = new_probs.view(old_probs.shape)

        # Compute the probability ratio between the old and new policy probabilities
        ratio = (new_probs - old_probs).exp()

        # Compute the unclipped surrogate objective
        unclipped_objective = ratio * advantages

        # Compute the clipped surrogate objective
        clipped_ratio = torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip)
        clipped_objective = clipped_ratio * advantages

        # Compute the minimum of the unclipped and clipped surrogate objectives
        surrogate_objective = -torch.min(unclipped_objective, clipped_objective).mean()

        return surrogate_objective, value_loss

    
    def learn(self):
        for i in range(self.n_epochs):
            if self.memory.mem_count < BATCH_SIZE:
                return

            states, actions, rewards, dones, probs, vals = self.memory.sample()
            states = torch.tensor(states , dtype=torch.float32).to(DEVICE)
            actions = torch.tensor(actions, dtype=torch.long).to(DEVICE)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
            dones = torch.tensor(dones, dtype=torch.float32).to(DEVICE)
            probs = torch.tensor(probs, dtype=torch.float32).exp().to(DEVICE)
            vals = torch.tensor(vals, dtype=torch.float32).to(DEVICE)
            batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

            # Compute the advantages for each experience
            advantages = self.compute_advantages(rewards, dones, vals)

            # Compute the new policy probabilities and value function estimates
            new_probs, new_values = self.compute_new_probs_and_values(states, actions)

            # Compute the clipped surrogate objective and value function loss
            clipped_objective, value_loss = self.compute_clipped_objective_and_value_loss(probs, new_probs, vals, new_values, advantages)

            # Compute the entropy bonus
            entropy_bonus = -self.entropy_weight * (new_probs * new_probs.log()).sum().mean()

            # Compute the total loss and take a gradient step
            total_loss = -(clipped_objective - value_loss + entropy_bonus)
            self.optimizer.zero_grad()
            total_loss.backward()

            # Add gradient clipping
            #clip_grad_norm_(self.actor.parameters(), 0.5)
            #clip_grad_norm_(self.critic.parameters(), 0.5)

            self.optimizer.step()


