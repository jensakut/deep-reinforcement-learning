from agent import Agent
from monitor import interact
import gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make('Taxi-v2')
agent = Agent(nA=6, alpha = 0.05, epsilon = 0.0001, gamma = 1)
avg_rewards, best_avg_reward = interact(env, agent)

plt.figure(figsize=(20, 10))
plt.plot(avg_rewards)
plt.show()