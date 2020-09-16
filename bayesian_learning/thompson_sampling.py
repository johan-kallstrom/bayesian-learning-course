# Implementation of the Thompson sampling algorithm

import numpy as np

from collections import deque

class ThompsonSampling:

    def __init__(self, priors):
        self.posteriors = priors
        self.n_arms = len(priors)

    def select_arm(self):
        draws = np.zeros(shape=(self.n_arms,))
        for i, posterior in enumerate(self.posteriors):
            theta = np.random.beta(a=posterior[0], b=posterior[1])
            draws[i] = theta
        arm = np.argmax(draws)
        return arm

    def learn(self, arm, reward):
        # Update alpha and beta in Beta distribution

        # alpha = alpha + reward
        self.posteriors[arm][0] += reward

        # beta = beta + abs(reward - 1)
        self.posteriors[arm][1] += abs(reward - 1)

    def plot_posteriors(self, save_to_file=False):
        pass

class SlidingWindowThompsonSampling:

    def __init__(self, priors, window_length=100):
        self.priors = priors
        self.n_arms = len(priors)
        self.recent_successes = [deque(maxlen=window_length) for _ in range(self.n_arms)]
        self.recent_failures = [deque(maxlen=window_length) for _ in range(self.n_arms)]

    def select_arm(self):
        draws = np.zeros(shape=(self.n_arms,))
        for i in range(self.n_arms):
            theta = np.random.beta(a=self.priors[i][0]+np.sum(self.recent_successes[i]), b=self.priors[i][1]+np.sum(self.recent_failures[i]))
            draws[i] = theta
        arm = np.argmax(draws)
        return arm

    def learn(self, arm, reward):
        # Update alpha and beta in Beta distribution

        # alpha = alpha + reward
        self.recent_successes.append(reward)

        # beta = beta + abs(reward - 1)
        self.recent_failures.append(abs(reward - 1))

    def plot_posteriors(self, save_to_file=False):
        pass