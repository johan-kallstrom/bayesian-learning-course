# Implementation of the Thompson sampling algorithm

import numpy as np

from scipy.stats import beta

class ThompsonSampling:

    def __init__(self, priors):
        self.posteriors = priors
        self.n_arms = len(priors)
        self.rng = np.random.default_rng()

    def select_arm(self):
        draws = np.zeros(shape=(self.n_arms,))
        for i, posterior in enumerate(self.posteriors):
            theta = self.rng.beta(a=posterior[0], b=posterior[1])
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
