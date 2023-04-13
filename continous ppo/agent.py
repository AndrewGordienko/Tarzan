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

from networks import actor_network, critic_network
from ppo_memory import PPOMemory

class PPOAgent:
    def __init__(self, env):
        # SETTINGS
        self.input_shape = env.observation_space.shape[0]
        self.n_actions = env.action_space.shape[0]

        self.env = env

        self.epochs = 4
        self.timesteps = 20
        self.mini_batch_size = 5
        self.gamma = 0.95
        self.tau = 0.95
        self.critic_coef = 0.5
        self.entropy_coef = 0.01
        self.epsilon = 0.2
        self.play_steps = self.timesteps
        
        self.adv_norm = False
        self.gae = False
        
        self.high_score = -np.inf
        self.avg_scores = []
        self.scores = []

        self.actor = actor_network(self.n_actions, self.input_shape)
        self.critic = critic_network(self.input_shape)
        
        self.device = self.actor.device
        
        self.memory = PPOMemory()

    def choose_action(self, state):
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action.detach().cpu().numpy()[0], value.item(), log_prob.detach().cpu().numpy()[0]


    def compute_gae(self, rewards, masks, values, next_val=None):
        # Initialize the returns list to an empty list
        returns = []
        
        # Initialize the GAE variable to zero
        gae = 0
        
        # If a next value is provided, append it to the values array
        value_ = np.append(values, next_val)

        # Iterate over the rewards and masks in reverse order
        for i in reversed(range(len(rewards))):
            # Compute the TD residual
            td_res = rewards[i] + self.gamma * value_[i+1] * masks[i] - value_[i]
            
            # Compute the GAE value
            gae = td_res + self.gamma * self.tau * masks[i] * gae
            
            # Compute the return value for the current time step and insert it at the beginning of the returns list
            returns.insert(0, gae + value_[i])

        # Convert the returns list to a PyTorch tensor and move it to the device
        return torch.tensor(returns).to(self.device)

    def compute_adv(self, rewards, masks, values, next_val=None):
        # Initialize the advantages array to zeros
        advantages = np.zeros(len(rewards), dtype=np.float32)
        
        # Initialize the GAE variable to zero
        gae = 0
        
        # If a next value is provided, append it to the values array
        value_ = np.append(values, next_val)

        # Iterate over the rewards and masks in reverse order
        for i in reversed(range(len(rewards))):
            # Compute the TD residual
            td_res = rewards[i] + self.gamma * value_[i+1] * masks[i] - value_[i]
            
            # Compute the GAE value
            gae = td_res + self.gamma * self.tau * masks[i] * gae
            
            # Set the advantage value for the current time step
            advantages[i] = gae

        # Convert the advantages array to a PyTorch tensor and move it to the device
        return torch.tensor(advantages).to(self.device)

    def learn(self, state_):
        # Estimate the value of the next state using the critic network
        next_val = self.critic(torch.tensor(state_).float().to(self.device)).detach().cpu().numpy().tolist()[0]

        # Train the actor and critic networks for multiple epochs
        for _ in range(self.epochs):
            # Get a batch of experiences from the memory buffer
            states, actions, rewards, values, log_probs, dones = self.memory.get_nps()

            # Compute the GAE or ADV values for the batch of experiences
            returns = self.compute_gae(rewards, dones, values, next_val=next_val)
            advantages = (returns.cpu() - values).flatten()

            # Convert the arrays to PyTorch tensors and move them to the device
            states = torch.tensor(states).float().to(self.device)
            actions = torch.tensor(actions).float().to(self.device)
            values = torch.tensor(values).float().to(self.device)
            log_probs = torch.tensor(log_probs).float().to(self.device)

            # Split the batch into mini-batches and train the networks on each mini-batch
            batches = self.memory.get_batches(self.mini_batch_size)

            for batch in batches:
                # Get the relevant tensors for the current mini-batch
                state = states[batch]
                action = actions[batch]
                old_log_prob = log_probs[batch]
                return_ = returns[batch].to(self.device)
                adv_ = advantages[batch].to(self.device)

                # Normalize the advantages for the current mini-batch
                adv_ = (adv_ - adv_.mean()) / (adv_.std() + 1e-8)

                # Compute the ratio of the new and old probabilities and the surrogate losses
                dist = self.actor(state)
                value_ = self.critic(state)
                new_log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().mean()
                ratio = (new_log_prob - old_log_prob).clamp(min=-10, max=10).exp()
                surr1 = ratio * adv_
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv_

                # Compute the actor and critic losses and the total loss
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value_).pow(2).mean() + 1e-8
                total_loss = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * entropy

                # Zero out the gradients, backpropagate the total loss, and update the network parameters
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
                self.actor.optimizer.step()
                self.critic.optimizer.step()
