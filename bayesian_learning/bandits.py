# Various types of multi-arm bandits

import numpy as np

class BernoulliBandit:

    def __init__(self, probs):
        self.probs = probs
        self.n_arms = len(probs)
        self.rng = np.random.default_rng()

    def draw(self, arm):
        assert arm < self.n_arms, "Arm outside range: %d" % self.n_arms
        reward = self.rng.binomial(n=1, p=self.probs[arm])
        expected_regret = np.max(self.probs) - self.probs[arm]
        return reward, expected_regret

class NonStationaryBernoulliBandit(BernoulliBandit):

    def __init__(self, probs):
        super(NonStationaryBernoulliBandit).__init__(probs)
        self.n_draws = 0

    def draw(self, arm):        
        reward = super.draw(arm)
        self.n_draws += 1
        self._update_probs(self.n_draws)
        return reward

    def _update_probs(self, n_draws):
        pass