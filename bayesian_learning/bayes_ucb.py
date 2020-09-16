# Implementation of the Bayes UCB algorithm.

import numpy as np

from scipy.stats import beta

class BayesUcb:

    def __init__(self, priors):
        self.posteriors = priors
        self.n_arms = len(priors)
        self.warmup_arm = 0
        self.time_step = 0
        self.rng = np.random.default_rng()

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