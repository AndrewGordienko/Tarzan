import numpy as np
import gymnasium as gym

env = gym.make('CartPole-v1', render_mode="human")
MEM_SIZE = 1000000
BATCH_SIZE = 32

class ReplayBuffer:
    def __init__(self):
        self.mem_count = 0
        
        self.states = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=np.bool_)
        self.probs = np.zeros(MEM_SIZE, dtype=np.float32)
        self.vals = np.zeros(MEM_SIZE, dtype=np.float32)
    
    def add(self, state, action, reward, done, probs, vals):
        mem_index = self.mem_count % MEM_SIZE
        
        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.dones[mem_index] =  done
        self.probs[mem_index] = probs
        self.vals[mem_index] = vals

        self.mem_count += 1
    
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)
        
        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        dones   = self.dones[batch_indices]
        probs = self.probs[batch_indices]
        vals = self.vals[batch_indices]

        return states, actions, rewards, dones, probs, vals