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
        self.input_shape = 4
        self.n_actions = 2

        self.env = env

        self.epochs = 4
        self.timesteps = 20
        self.mini_batch_size = 5
        self.gamma = 0.99
        self.tau = 0.95
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
        dist = self.actor.forward(state)
        #print(dist)
        value = self.critic.forward(state)
        action = dist.sample()
        
        prob = dist.log_prob(action)
        #print(prob.shape)
#
        return action.detach().cpu().numpy()[0], value.item(), prob.detach().cpu().numpy()[0]

    def compute_gae(rewards, masks, values, next_val=None, gamma=0.99, tau=0.95):
        """
        Compute GAE values for a sequence of rewards, masks, and estimated values.

        Args:
            rewards (numpy array): A sequence of rewards.
            masks (numpy array): A sequence of masks indicating whether an episode has ended.
            values (numpy array): A sequence of estimated state values.
            next_val (float, optional): The estimated value of the next state. Defaults to None.
            gamma (float, optional): The discount factor. Defaults to 0.99.
            tau (float, optional): The GAE parameter. Defaults to 0.95.

        Returns:
            numpy array: A sequence of GAE values.
        """
        returns = []  # Initialize the list to hold the GAE returns
        gae = 0  # Initialize the GAE value to 0
        value_ = np.append(values, next_val)  # Append the estimated value of the next state (if provided) to the list of values

        for i in reversed(range(len(rewards))):  # Iterate over the rewards in reverse order
            td_res = rewards[i] + gamma * value_[i+1] * masks[i] - value_[i]  # Compute the TD residual
            gae = td_res + gamma * tau * masks[i] * gae  # Update the GAE value
            returns.insert(0, gae + value_[i])  # Insert the GAE return into the list of returns

        return np.array(returns)  # Convert the list of returns to a numpy array and return it

    def compute_adv(rewards, masks, values, next_val=None, gamma=0.99, tau=0.95):
        """
        Compute Advantage values for a sequence of rewards, masks, and estimated values.

        Args:
            rewards (numpy array): A sequence of rewards.
            masks (numpy array): A sequence of masks indicating whether an episode has ended.
            values (numpy array): A sequence of estimated state values.
            next_val (float, optional): The estimated value of the next state. Defaults to None.
            gamma (float, optional): The discount factor. Defaults to 0.99.
            tau (float, optional): The GAE parameter. Defaults to 0.95.

        Returns:
            numpy array: A sequence of Advantage values.
        """
        advantages = np.zeros(len(rewards), dtype=np.float32)  # Initialize the array to hold the Advantage values
        gae = 0  # Initialize the GAE value to 0
        value_ = np.append(values, next_val)  # Append the estimated value of the next state (if provided) to the list of values

        for i in reversed(range(len(rewards))):  # Iterate over the rewards in reverse order
            td_res = rewards[i] + gamma * value_[i+1] * masks[i] - value_[i]  # Compute the TD residual
            gae = td_res + gamma * tau * masks[i] * gae  # Update the GAE value
            advantages[i] = gae  # Store the GAE value as the Advantage value for this timestep

        return advantages  # Return the array of Advantage values


    def learn(self, state_):
        # Estimate the value of the next state using the critic network
        next_val = self.critic.forward(torch.tensor(state_).float().to(self.device)).detach().cpu().numpy().tolist()[0]

        # Train the actor and critic networks for multiple epochs
        for _ in range(self.epochs):
            # Get a batch of experiences from the memory buffer
            states, actions, rewards, values, probs, dones = self.memory.get_nps()
            
            # Compute the GAE or ADV values for the batch of experiences
            if self.gae:
                returns =  self.compute_gae(rewards, dones, values, next_val=next_val)
            else:
                advantages = self.compute_adv(rewards, dones, values, next_val=next_val)

            # Convert the arrays to PyTorch tensors and move them to the device
            probs   =  torch.tensor(probs  ).reshape(self.play_steps).detach().to(self.device)
            states  =  torch.tensor(states ).float().to(self.device)
            actions =  torch.tensor(actions).reshape(self.play_steps).detach().to(self.device)
            values  =  torch.tensor(values ).reshape(self.play_steps).detach().to(self.device)
            dones   =  torch.tensor(dones  ).to(self.device)

            # If using GAE, compute the ADV values from the GAE returns
            if self.gae:
                advantages = returns - values
            else: 
                returns = advantages + values

            # Convert the arrays to PyTorch tensors and move them to the device
            returns =  returns.reshape(self.play_steps).detach().to(self.device)

            # Split the batch into mini-batches and train the networks on each mini-batch
            batches = self.memory.get_batches(self.mini_batch_size)
            for batch in batches:
                # Get the relevant tensors for the current mini-batch
                old_log_probs =      probs[batch]
                state         =     states[batch]
                action        =    actions[batch]
                return_       =    returns[batch]
                adv_          = advantages[batch].reshape(self.mini_batch_size, 1)
                
                # Normalize the ADV values if specified
                if self.adv_norm:
                    adv_ = (adv_ - adv_.mean()) / (adv_.std() + 1e-4)

                # Compute the ratio of the new and old probabilities and the surrogate losses
                epsilon = 0.2
                dist   = self.actor.forward(state)
                value_ = self.critic.forward(state)
                new_log_probs = dist.log_prob(action)
                entropy = dist.entropy().mean()
                ratio = new_log_probs.exp() / old_log_probs.exp()
                surr1 = ratio * adv_
                surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * adv_

                # Compute the actor and critic losses and the total loss
                a_loss = -torch.min(surr1, surr2).mean()
                c_loss = (return_ - value_)**2
                c_loss = c_loss.mean()
                total_loss = a_loss + 0.25*c_loss# - 0.001*entropy
                    
                # Zero out the gradients, backpropagate the total loss, and update the network parameters
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
