import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import gym
import numpy as np
import warnings
from networks import actor_network, critic_network
from ppo_memory import PPOMemory

warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda")


class TRPOAgent:
    def __init__(self, env):
        # SETTINGS
        self.input_shape = env.observation_space.shape[0]
        self.n_actions = env.action_space.shape[0]

        self.env = env
        self.epochs = 10
        self.timesteps = 20
        self.mini_batch_size = 5
        self.gamma = 0.99
        self.tau = 0.95
        self.critic_coef = 0.5
        self.entropy_coef = 0.01
        self.max_kl = 0.01
        self.damping = 0.1
        self.play_steps = self.timesteps

        self.high_score = -np.inf
        self.avg_scores = []
        self.scores = []

        self.actor = actor_network(self.n_actions, self.input_shape)
        self.critic = critic_network(self.input_shape)

        self.device = self.actor.device

        self.memory = PPOMemory()

        self.actor_optimizer = Adam(self.actor.parameters())
        self.critic_optimizer = Adam(self.critic.parameters())

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

    def conjugate_gradient(self, Fvp, b, nsteps=10, residual_tol=1e-6):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)

        for _ in range(nsteps):
            Fvp_x = Fvp(p)
            alpha = rdotr / torch.dot(p, Fvp_x)
            if torch.isnan(alpha):
                alpha = torch.tensor(1e-4, device=self.device)
            x += alpha * p
            r -= alpha * Fvp_x
            new_rdotr = torch.dot(r, r)
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        print(x)
        return x


    def linesearch(self, f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=0.1):
        with torch.no_grad():
            max_backtracks = int(max_backtracks)
            fval = f(x)
            for stepfrac in [0.5 ** i for i in range(int(max_backtracks))]:  # Use list comprehension
                xnew = x + stepfrac * fullstep
                newfval = f(xnew)
                actual_improve = fval - newfval
                expected_improve = expected_improve_rate * stepfrac
                ratio = actual_improve / expected_improve

                if ratio.item() > accept_ratio:
                    return xnew
        return x


    def Fvp(self, v, states, actions, old_policy):
        v = v.detach()
        states = states.requires_grad_(True)
        actions = actions.requires_grad_(True)
        kl = torch.mean(old_policy.log_prob(actions) * (old_policy.log_prob(actions) - self.actor(states).log_prob(actions)).detach())
        kl = kl.sum()
        grads = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True, allow_unused=True)
        
        non_empty_grads = [grad for grad in grads if grad is not None]
        if not non_empty_grads:
            return torch.zeros_like(v)
        
        flat_grad_kl = torch.cat([grad.view(-1) for grad in non_empty_grads])

        kl_v = (flat_grad_kl * v).sum()
        grads_v = torch.autograd.grad(kl_v, self.actor.parameters(), allow_unused=True)
        flat_grad_kl_v = torch.cat([grad.contiguous().view(-1) for grad in grads_v if grad is not None]).data

        return flat_grad_kl_v + v * self.damping


    def get_loss(self, states, actions, advantages, old_action_log_probs, device):
        states = states.to(device)
        actions = actions.to(device)
        advantages = advantages.to(device)
        old_action_log_probs = old_action_log_probs.to(device)

        policy = self.actor(states)
        action_log_probs = policy.log_prob(actions).sum(dim=-1, keepdim=True)
        return (-advantages * torch.exp(action_log_probs - old_action_log_probs.detach())).mean()
    
    def get_params(self):
        return torch.cat([param.view(-1) for param in self.actor.parameters()])

    def set_params(self, x):
        idx = 0
        for param in self.actor.parameters():
            numel = param.numel()
            param.data.copy_(x[idx:idx+numel].view_as(param))
            idx += numel
    
    def learn(self, state_):
        # Estimate the value of the next state using the critic network
        next_val = self.critic(torch.tensor(state_).float().to(self.device)).detach().cpu().numpy().tolist()[0]

        # Get a batch of experiences from the memory buffer
        states, actions, rewards, values, log_probs, dones = self.memory.get_nps()

        # Compute the GAE or ADV values for the batch of experiences
        returns = self.compute_gae(rewards, dones, values, next_val=next_val)
        advantages = (returns.cpu() - values).flatten()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert the arrays to PyTorch tensors and move them to the device
        states = torch.tensor(states).float().to(self.device)
        actions = torch.tensor(actions).float().to(self.device)
        values = torch.tensor(values).float().to(self.device)
        log_probs = torch.tensor(log_probs).float().to(self.device)
        
        # Make sure returns tensor has the correct dtype
        returns = returns.float().to(self.device)

        # Train the critic network
        for _ in range(self.epochs):
            value_preds = self.critic(states)
            value_loss = F.mse_loss(value_preds, returns)
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

        # Update the actor network using TRPO
        with torch.no_grad():
            old_policy = self.actor(states)
            old_action_log_probs = old_policy.log_prob(actions).sum(dim=-1, keepdim=True)

        loss_before = self.get_loss(states, actions, advantages, old_action_log_probs, self.device)
        grads = torch.autograd.grad(loss_before, self.actor.parameters())
        grads_flat = torch.cat([grad.view(-1) for grad in grads]).detach()

        Fvp = lambda v: self.Fvp(v, states, actions, old_policy)
        get_loss = lambda: self.get_loss(states, actions, advantages, old_action_log_probs)
        fullstep = self.conjugate_gradient(Fvp, grads_flat)
        shs = 0.5 * (fullstep * Fvp(fullstep)).sum()
        lm = torch.sqrt(shs / (self.max_kl + 1e-8)) # Add small constant to prevent
        fullstep = fullstep / lm.item()

        neggdotstepdir = (-grads_flat * fullstep).sum()
        prev_params = self.get_params()
        new_params = self.linesearch(lambda x: self.get_loss(states, actions, advantages, old_action_log_probs, x), prev_params, fullstep, neggdotstepdir, lm.item())
        self.set_params(new_params)

        # Clear the memory buffer
        self.memory.clear_memory()

        
