# Implementation of the Thompson sampling algorithm

import numpy as np

from collections import deque

class ThompsonSampling:

    def __init__(self, priors):
        self.priors = priors.copy()
        self.posteriors = priors.copy()
        self.n_arms = len(priors)

    def select_arm(self):
        draws = np.zeros(shape=(self.n_arms,))
        for i in range(self.posteriors.shape[0]):
            theta = np.random.beta(a=self.posteriors[i,0], b=self.posteriors[i,1])
            draws[i] = theta
        arm = np.argmax(draws)
        # print("Arm:", arm)
        return arm

    def learn(self, arm, reward):
        # Update alpha and beta in Beta distribution

        # alpha = alpha + reward
        self.posteriors[arm][0] += reward

        # beta = beta + abs(reward - 1)
        self.posteriors[arm][1] += abs(reward - 1)

    def reset(self):
        self.posteriors = self.priors.copy()

class SlidingWindowThompsonSampling:

    def __init__(self, priors, window_length=2000):
        self.priors = priors.copy()
        self.n_arms = len(priors)
        self.window_length = window_length
        self.reset()

    def select_arm(self):
        draws = np.zeros(shape=(self.n_arms,))
        for i in range(self.n_arms):
            theta = np.random.beta(a=self.priors[i][0]+np.sum(self.recent_rewards[i]), b=self.priors[i][1]+len(self.recent_rewards[i])-np.sum(self.recent_rewards[i]))
            draws[i] = theta
        arm = np.argmax(draws)
        return arm

    def learn(self, arm, reward):
        # Store recent rewards
        self.recent_rewards[arm].append(reward)

    def reset(self):
        self.recent_rewards = [deque(maxlen=self.window_length) for _ in range(self.n_arms)]
