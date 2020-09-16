# Various types of multi-arm bandits

import numpy as np

class BernoulliBandit:

    def __init__(self, probs):
        self.probs = probs
        self.initial_probs = probs
        self.n_arms = len(probs)

    def draw(self, arm):
        assert arm < self.n_arms, "Arm outside range: %d" % self.n_arms
        reward = np.random.binomial(n=1, p=self.probs[arm])
        expected_regret = np.max(self.probs) - self.probs[arm]
        return reward, expected_regret

class NonStationaryBernoulliBandit(BernoulliBandit):

    def __init__(self, probs, total_draws):
        assert 3 == self.n_arms, "Currently 3 arms is expected"
        super(NonStationaryBernoulliBandit).__init__(probs)
        self.n_draws = 0
        self.total_draws = total_draws

    def draw(self, arm):        
        reward = super.draw(arm)
        self.n_draws += 1
        self._update_probs(self.n_draws)
        return reward

    def _update_probs(self, n_draws):
        self.probs[0] = self.probs[0] + (n_draws / self.total_draws) * (self.probs[2] - self.probs[0])
        self.probs[2] = self.probs[2] + (n_draws / self.total_draws) * (self.probs[0] - self.probs[2])