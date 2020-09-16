# Implementation of the Upper Confidence Bound (UCB) algorithm

import numpy as np

class Ucb:

    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.warmup_arm = 0
        self.time_step = 0
        self.rewards = np.zeros(shape=(n_arms,))
        self.n_slected = np.zeros(shape=(n_arms,))

    def select_arm(self):
        if self.warmup_arm < self.n_arms:
            arm = self.warmup_arm
            self.warmup_arm += 1
        else:
            ucb_estimates = np.zeros(shape=(self.n_arms,))
            for i in range(self.n_arms):
                ucb_term = np.sqrt((2 * np.log(self.time_step)) / self.n_slected[i])
                ucb_estimates[i] = self.rewards[i] / self.n_slected[i] + ucb_term
            arm = np.argmax(ucb_estimates)
        self.time_step += 1
        return arm

    def learn(self, arm, reward):
        self.n_slected[arm] += 1
        self.rewards[arm] += reward        
