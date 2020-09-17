# Various types of multi-arm bandits

import numpy as np

class BernoulliBandit:

    def __init__(self, probs):
        self.probs = probs
        self.initial_probs = probs.copy()
        self.n_arms = len(probs)

    def draw(self, arm):
        assert arm < self.n_arms, "Arm outside range: %d" % self.n_arms
        reward = np.random.binomial(n=1, p=self.probs[arm])
        expected_regret = np.max(self.probs) - self.probs[arm]
        return reward, expected_regret

class NonStationaryBernoulliBandit(BernoulliBandit):

    def __init__(self, probs, total_draws):
        super().__init__(probs)
        assert 3 == self.n_arms, "Currently 3 arms is expected"
        self.n_draws = 0
        self.total_draws = total_draws

    def draw(self, arm):        
        reward = super().draw(arm)
        return reward

    def _update_probs(self, n_draws):
        self.probs[0] = self.initial_probs[0] + (n_draws / self.total_draws) * (self.initial_probs[2] - self.initial_probs[0])
        self.probs[2] = self.initial_probs[2] + (n_draws / self.total_draws) * (self.initial_probs[0] - self.initial_probs[2])
        # print((self.initial_probs[2] - self.initial_probs[0]))
        # print((self.initial_probs[0] - self.initial_probs[2]))
        # print((n_draws / self.total_draws))
        # print(self.probs)