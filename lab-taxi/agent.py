import random

import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha = 0.01, epsilon = 0.005, gamma = 1):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.i_episode = 0
        print('alpha: ', alpha, 'gamma: ', gamma, 'epsilon ', epsilon)
    def select_action(self, state):
        """ Given the state, select an action. 
        Uses epsilon-greedy policy


        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if random.random() > self.epsilon:
            return np.argmax(self.Q[state])
        else:
            return random.choice(np.arange(self.nA))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Use expected sarsa
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        Q_sa = self.Q[state][action]
        # build a greedy policy with respect to epsilon
        policy_s = np.ones(self.nA) * self.epsilon / self.nA
        policy_s[np.argmax(self.Q[next_state])] = 1 - self.epsilon + (self.epsilon / self.nA)
        # get value of state at next timestep
        Q_sa_next = np.dot(self.Q[next_state], policy_s)

        # construct target
        target = reward + self.gamma * Q_sa_next
        # return updated value
        self.Q[state][action] = Q_sa + self.alpha * (target - Q_sa)

        # count the episodes internally
        if done:
            self.i_episode += 1
            # # decay epsilon
            self.epsilon = 1 / self.i_episode