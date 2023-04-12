import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import gym
import numpy as np
import warnings; warnings.filterwarnings('ignore')
from matplotlib.animation import FuncAnimation

from agent import PPOAgent

# Create the CartPole environment and the PPO agent
env = gym.make('CartPole-v1').unwrapped
agent = PPOAgent(env)

# Initialize variables for tracking the score and episode returns
frame = 0
high_score = -np.inf
returns = None
episode = 0

# Start the main training loop
while True:
    # Reset the environment and initialize the state
    state = env.reset()
    done = False
    score = 0

    # Run the episode until completion
    while not done:
        # Choose an action based on the current state
        action, value, prob = agent.choose_action(state)
        # Take the chosen action and observe the new state, reward, and done flag
        state_, reward, done, info = env.step(action)
        # Update the episode score
        score += reward
        # Store the experience in the memory buffer
        agent.memory.store_memory(state, action, reward, value, prob, 1-done)
        
        # Render the environment if we've completed enough episodes
        if episode >= 50:
            env.render()

        # Update the state and frame counter
        state = state_
        frame += 1

        # If we've completed enough frames, train the agent
        if frame % agent.play_steps == 0:
            agent.learn(state_)
            agent.memory.reset()

    # Update the running list of scores and compute the average score
    agent.scores.append(score)
    agent.avg_scores.append(np.mean(agent.scores))
    high_score = max(high_score, score)
    avg = np.mean(agent.scores)
    episode += 1

    # Print the episode statistics
    print(f'episode: {episode}, high_score: {high_score}, score: {score}, avg: {avg}')

# Show the final score plot
plt.show()
