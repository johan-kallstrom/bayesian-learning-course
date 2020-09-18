# Run experiment 1

from bayesian_learning.bandits import BernoulliBandit
from bayesian_learning.thompson_sampling import ThompsonSampling
from bayesian_learning.bayes_ucb import BayesUcb
from bayesian_learning.ucb import Ucb
from bayesian_learning.q_learning import EpsilonGreedySampling

import numpy as np

from matplotlib import pyplot as plt

import pickle

# Experiment settings
np.random.seed(2020)
n_runs = 100
n_draws = 10000
posteriors_to_save = [0,9,49,99,499,999,1999,3999,7999]
bandit_probs = np.array([0.8, 0.5, 0.2])
bernoulli_bandit = BernoulliBandit(probs=bandit_probs)

# Priors
uninformed_priors = np.array([[1.0, 1.0],
                             [1.0, 1.0],
                             [1.0, 1.0]])

informed_priors = np.array([[5.0, 5.0],  # medium
                           [2.0, 8.0],   # hard
                           [1.0, 10.0]]) # easy

# Players
players = []

thompson_sampler = ThompsonSampling(priors=uninformed_priors)
players.append([thompson_sampler,['Thompson Sampling']])

informed_thompson_sampler = ThompsonSampling(priors=informed_priors)
players.append([informed_thompson_sampler,['Thompson Sampling with Informed Prior']])

bayes_ucb = BayesUcb(priors=uninformed_priors)
players.append([bayes_ucb,['Bayes UCB']])

ucb = Ucb(n_arms=bandit_probs.shape[0])
players.append([ucb,['UCB']])

epsilon_greedy_sampling = EpsilonGreedySampling(n_arms=bandit_probs.shape[0])
players.append([epsilon_greedy_sampling,['Epsilon Greedy']])

# Run the experiment
n_players = len(players)
cumulative_reward = np.zeros(shape=(1, n_players, n_runs))
cumulative_reward_history = np.zeros(shape=(n_draws, n_players, n_runs))
cumulative_regret = np.zeros(shape=(1, n_players, n_runs))
cumulative_regret_history = np.zeros(shape=(n_draws, n_players, n_runs))
posterior_estimates = np.zeros(shape=(9,3,2,n_runs))
for run in range(n_runs):
    print("Run: ", run+1)
    for player in players:
        player[0].reset()
    post_idx = 0
    for draw in range(n_draws):
        for i, player in enumerate(players):
            # Draw and learn
            arm = player[0].select_arm()
            reward, expected_regret = bernoulli_bandit.draw(arm)
            player[0].learn(arm, reward)

            # Calculate metrics
            cumulative_reward[:,i,run] += reward
            cumulative_reward_history[draw, i, run] = cumulative_reward[:,i,run]
            cumulative_regret[:,i,run] += expected_regret
            cumulative_regret_history[draw, i, run] = cumulative_regret[:,i,run]
        if draw in posteriors_to_save:
            posterior_estimates[post_idx,:,:,run] = thompson_sampler.posteriors
            post_idx += 1

posterior_estimates = np.mean(posterior_estimates, axis=3)
pickle.dump(posterior_estimates, open("./experiments/posterior_estimates.p", "wb"))
pickle.dump(posteriors_to_save, open("./experiments/posterior_info.p", "wb"))

mean_regret = np.mean(cumulative_regret_history,axis=2)
pickle.dump(mean_regret, open("./experiments/mean_regret.p", "wb"))
for i in range(n_players):    
    plt.plot(mean_regret[:,i], label=players[i][1][0])

plt.xlabel('draw')
plt.ylabel('cumulative regret')
plt.title('Comparison of Bayesian and Frequentist Algorithms')
plt.legend()
plt.savefig("Experiment1_Regret.pdf", bbox_inches='tight')
plt.show()
