import math
import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
import numpy 
import torch

from agent import Agent

env = gym.make('CartPole-v1', render_mode="human")

EPISODES = 501
MEM_SIZE = 1000000
BATCH_SIZE = 32

best_reward = float("-inf")
total_reward = 0
episode_number = []
average_reward_number = []

agent = Agent(n_actions=env.action_space.n, input_dims=env.observation_space.shape)

for i in range(1, EPISODES):
    observation, info = env.reset()
    score = 0
    done = False
    step = 0

    while not done:
        action, prob, val = agent.choose_action(observation)
        observation_, reward, done, truncated, info = env.step(action)
        agent.remember(observation, action, reward, done, prob, val)

        score += reward
        step += 1
        if step % 5 == 0: agent.learn()
        observation = observation_

    total_reward += score
    print(f"episode {i} average reward {total_reward/i} score {score}")


        