# Implementation of the Q-learning algorithm.

import numpy as np

class EpsilonGreedySampling:

    def __init__(self, n_arms, epsilon=0.15):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.time_step = 0
        self.rewards = np.zeros(shape=(n_arms,))
        self.n_slected = np.zeros(shape=(n_arms,))

    def select_arm(self):
        if np.random.random_sample() < self.epsilon:
            arm = np.random.randint(0, self.n_arms - 1)
        else:
            means = np.zeros(shape=(self.n_arms,))            
            for i in range(self.n_arms):
                if self.n_slected[i] > 0:
                    means[i] = self.rewards[i] / self.n_slected[i]
                else:
                    means[i] = 0
            arm = np.argmax(means)
        return arm

    def learn(self, arm, reward):
        self.n_slected[arm] += 1
        self.rewards[arm] += reward