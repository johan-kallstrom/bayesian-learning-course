# Run experiment 1

from bayesian_learning.bandits import BernoulliBandit
from bayesian_learning.thompson_sampling import ThompsonSampling
from bayesian_learning.bayes_ucb import BayesUcb
from bayesian_learning.ucb import Ucb
from bayesian_learning.q_learning import EpsilonGreedySampling

import numpy as np

from matplotlib import pyplot as plt

# Experiment settings
np.random.seed(1976)
n_draws = 10000
bandit_probs = np.array([0.5, 0.3, 0.2])
bernoulli_bandit = BernoulliBandit(probs=bandit_probs)

# Priors
uninformed_priors = [np.array([1.0, 1.0]),
                     np.array([1.0, 1.0]),
                     np.array([1.0, 1.0])]

informed_priors = [np.array([5.0, 5.0]),  # easy
                   np.array([2.0, 8.0]),  # medium
                   np.array([1.0, 10.0])] # hard

# Players
players = []

thompson_sampler = ThompsonSampling(priors=uninformed_priors)
players.append(thompson_sampler)

thompson_sampler = ThompsonSampling(priors=informed_priors)
players.append(thompson_sampler)

bayes_ucb = BayesUcb(priors=uninformed_priors)
players.append(bayes_ucb)

ucb = Ucb(n_arms=3)
players.append(ucb)

epsilon_greedy_sampling = EpsilonGreedySampling(n_arms=3)
players.append(epsilon_greedy_sampling)


# Run the experiment
n_players = len(players)
cumulative_reward = np.zeros(shape=(1, n_players))
cumulative_reward_history = np.zeros(shape=(n_draws, n_players))
cumulative_regret = np.zeros(shape=(1, n_players))
cumulative_regret_history = np.zeros(shape=(n_draws, n_players))
for draw in range(n_draws):
    for i, player in enumerate(players):
        # Draw and learn
        arm = player.select_arm()
        reward, expected_regret = bernoulli_bandit.draw(arm)
        player.learn(arm, reward)

        # Calculate metrics
        cumulative_reward[:,i] += reward
        cumulative_reward_history[draw, i] = cumulative_reward[:,i]
        cumulative_regret[:,i] += expected_regret
        cumulative_regret_history[draw, i] = cumulative_regret[:,i]

colors = []
for i in range(n_players):
    plt.plot(cumulative_regret_history[:,i])
plt.show()