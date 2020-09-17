# Run experiment 2

from bayesian_learning.bandits import NonStationaryBernoulliBandit
from bayesian_learning.thompson_sampling import ThompsonSampling, SlidingWindowThompsonSampling
from bayesian_learning.bayes_ucb import BayesUcb, SlidingWindowBayesUcb
from bayesian_learning.ucb import Ucb
from bayesian_learning.q_learning import EpsilonGreedySampling

import numpy as np

from matplotlib import pyplot as plt

import pickle

# Experiment settings
np.random.seed(2020)
n_runs = 100
n_draws = 10000
bandit_probs = np.array([0.8, 0.5, 0.2])
bernoulli_bandit = NonStationaryBernoulliBandit(probs=bandit_probs, total_draws=n_draws)

# Priors
uninformed_priors = np.array([[1.0, 1.0],
                             [1.0, 1.0],
                             [1.0, 1.0]])

# Players
players = []

thompson_sampler = ThompsonSampling(priors=uninformed_priors)
players.append([thompson_sampler,['Thompson Sampling']])

sliding_window_thompson_sampler = SlidingWindowThompsonSampling(priors=uninformed_priors)
players.append([sliding_window_thompson_sampler,['Sliding Window Thompson Sampling']])

bayes_ucb = BayesUcb(priors=uninformed_priors)
players.append([bayes_ucb,['Bayes UCB']])

sliding_window_bayes_ucb = SlidingWindowBayesUcb(priors=uninformed_priors)
players.append([sliding_window_bayes_ucb,['Sliding Window Bayes UCB']])

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
        bernoulli_bandit._update_probs(draw)
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

mean_regret = np.mean(cumulative_regret_history,axis=2)
pickle.dump(mean_regret, open("./experiments/experiment_2_mean_regret_long.p", "wb"))
for i in range(n_players):    
    plt.plot(mean_regret[:,i], label=players[i][1][0])

plt.xlabel('draw')
plt.ylabel('cumulative regret')
plt.title('Comparison of Bayesian and Frequentist Algorithms')
plt.legend()
plt.savefig("Experiment2_Regret_long.pdf", bbox_inches='tight')
# plt.show()


# Run experiment to see regret as function of window sizee
n_players = 2
n_runs = 100
window_lengths = [10, 50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000]

regrets = np.zeros(shape=(n_players,len(window_lengths)))
for w_idx, window_length in enumerate(window_lengths):
    cumulative_regret = np.zeros(shape=(n_players, n_runs))
    for run in range(n_runs):
        print("Run: ", run+1)
        players = []

        sliding_window_thompson_sampler = SlidingWindowThompsonSampling(priors=uninformed_priors, window_length=window_length)
        players.append([sliding_window_thompson_sampler,['Thompson Sampling']])

        sliding_window_bayes_ucb = SlidingWindowBayesUcb(priors=uninformed_priors, window_length=window_length)
        players.append([sliding_window_bayes_ucb,['Bayes UCB']])
        post_idx = 0
        for draw in range(n_draws):
            bernoulli_bandit._update_probs(draw)
            for i, player in enumerate(players):
                # Draw and learn
                arm = player[0].select_arm()
                reward, expected_regret = bernoulli_bandit.draw(arm)
                player[0].learn(arm, reward)

                # Calculate metrics
                cumulative_regret[i,run] += expected_regret
    mean_regret = np.mean(cumulative_regret,axis=1)
    for j in range(n_players):  
        regrets[j,w_idx] = mean_regret[j]

pickle.dump(regrets, open("./experiments/experiment_2_regretes_by_window_long.p", "wb"))

plt.clf()
for i in range(n_players):    
    plt.plot(window_lengths, regrets[i,:], label=players[i][1][0])

plt.xlabel('window length')
plt.ylabel('cumulative regret')
plt.title('Effect of Window Length on Regret')
plt.legend()
plt.savefig("Experiment2_Regret_by_Window_long.pdf", bbox_inches='tight')
plt.show()