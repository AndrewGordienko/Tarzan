import numpy as np
import gym

env = gym.make('LunarLanderContinuous-v2')
#env = gym.make('BipedalWalker-v3')
MEM_SIZE = 1000000
BATCH_SIZE = 32

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

        self.value = []
        self.probs = []
        self.dones = []

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []

        self.value = []
        self.probs = []
        self.dones = []

    def store_memory(self, s, a, r, v, p, d):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

        self.value.append(v)
        self.probs.append(p)
        self.dones.append(d)

    def get_batches(self, batch_size):
        t_l = len(self.dones)
        indices = np.arange(t_l, dtype=np.float32)
        np.random.shuffle(indices)
        start_indicies = np.arange(0, t_l, batch_size)
        batches = [indices[i:i+batch_size] for i in start_indicies]

        return batches

    def get_nps(self):
        return np.array(self.states), \
                np.array(self.actions), \
                np.array(self.rewards), \
                np.array(self.value), \
                np.array(self.probs), \
                np.array(self.dones) 