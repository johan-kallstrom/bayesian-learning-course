# Implementation of the Bayes UCB algorithm.

import numpy as np

from collections import deque 
from scipy.stats import beta

class BayesUcb:

    def __init__(self, priors):
        self.priors = priors.copy()
        self.n_arms = len(priors)
        self.reset()

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

    def reset(self):
        self.posteriors = self.priors.copy()
        self.warmup_arm = 0
        self.time_step = 0

class SlidingWindowBayesUcb:

    def __init__(self, priors, window_length=2000):
        self.priors = priors.copy()
        self.n_arms = len(priors)
        self.window_length = window_length
        self.reset()

    def select_arm(self):
        if self.warmup_arm < self.n_arms:
            arm = self.warmup_arm
            self.warmup_arm += 1
        else:
            qs = np.zeros(shape=(self.n_arms,))
            for i in range(self.n_arms):
                p = 1 - 1 / self.time_step
                q = beta.ppf(p, a=self.priors[i][0]+np.sum(self.recent_rewards[i]), b=self.priors[i][1]+len(self.recent_rewards[i])-np.sum(self.recent_rewards[i]), loc=0, scale=1)
                qs[i] = q
            arm = np.argmax(qs)
        self.time_step += 1
        return arm

    def learn(self, arm, reward):
        # Store recent rewards
        self.recent_rewards[arm].append(reward)

    def reset(self):
        self.posteriors = self.priors.copy()
        self.warmup_arm = 0
        self.time_step = 0
        self.recent_rewards = [deque(maxlen=self.window_length) for _ in range(self.n_arms)]