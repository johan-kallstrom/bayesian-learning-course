from scipy.stats import beta

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import pickle

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

posterior_estimates = pickle.load(open( "./experiments/posterior_estimates.p", "rb"))
posterior_info = pickle.load(open( "./experiments/posterior_info.p", "rb"))

colors = ['b-', 'r-', 'g-']
for i in range(posterior_estimates.shape[0]):
    ax = plt.subplot(3,3,(i+1))
    for arm in range(3):
        a = posterior_estimates[i,arm,0]
        b = posterior_estimates[i,arm,1]
        x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
        # print("After " + str(posterior_info[i] + 1) + " Draws")
        ax.plot(x, beta.pdf(x, a, b), colors[arm], lw=5, alpha=0.6, label='arm_'+str(arm+1))
        ax.set_title("After " + str(posterior_info[i] + 1) + " Draws")
    # ax.set_ylim([0,8])
    ax.set_xlim([0,1])

plt.legend()
plt.savefig("Experiment1_Densities.pdf", bbox_inches='tight')
plt.show()

