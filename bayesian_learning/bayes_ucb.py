# Implementation of the Bayes UCB algorithm.

import numpy as np

from collections import deque 
from scipy.stats import beta

class BayesUcb:

    def __init__(self, priors):
        self.posteriors = priors
        self.n_arms = len(priors)
        self.warmup_arm = 0
        self.time_step = 0
        # self.rng = np.random.default_rng()

    def select_arm(self):
        if self.warmup_arm < self.n_arms:
            arm = self.warmup_arm
            self.warmup_arm += 1
        else:
            qs = np.zeros(shape=(self.n_arms,))
            for i, posterior in enumerate(self.posteriors):
                p = 1 - 1 / self.time_step
                q = beta.ppf(p, a=posterior[0], b=posterior[1], loc =0, scale = 1)
                qs[i] = q
            arm = np.argmax(qs)
        self.time_step += 1
        return arm

    def learn(self, arm, reward):
        # Update alpha and beta in Beta distribution

        # alpha = alpha + reward
        self.posteriors[arm][0] += reward

        # beta = beta + abs(reward - 1)
        self.posteriors[arm][1] += abs(reward - 1)

class SlidingWindowBayesUcb:

    def __init__(self, priors, window_length=100):
        self.priors = priors
        self.n_arms = len(priors)
        self.warmup_arm = 0
        self.time_step = 0
        self.recent_successes = [deque(maxlen=window_length) for _ in range(self.n_arms)]
        self.recent_failures = [deque(maxlen=window_length) for _ in range(self.n_arms)]

    def select_arm(self):
        if self.warmup_arm < self.n_arms:
            arm = self.warmup_arm
            self.warmup_arm += 1
        else:
            qs = np.zeros(shape=(self.n_arms,))
            for i in range(self.n_arms):
                p = 1 - 1 / self.time_step
                q = beta.ppf(p, a=self.priors[i][0]+np.sum(self.recent_successes[i]), b=self.priors[i][1]+np.sum(self.recent_failures[i]), loc=0, scale=1)
                qs[i] = q
            arm = np.argmax(qs)
        self.time_step += 1
        return arm

    def learn(self, arm, reward):
        # Update alpha and beta in Beta distribution

        # alpha = alpha + reward
        self.recent_successes.append(reward)

        # beta = beta + abs(reward - 1)
        self.recent_failures.append(abs(reward - 1))