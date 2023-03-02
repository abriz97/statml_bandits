import numpy as np
import matplotlib.pyplot as plt

from src.agents import *
from src.bandits import *


def run(n, agent, bandit):
    regret = 0 # tracks cumulative regret
    output = [regret] # stores cumulative regrets

    for t in range(1, n + 1): 
        action = agent.policy(t)
        reward = bandit.pull(action)
        agent.update(action, reward)

        # update and store the cumulative regret
        regret += bandit.regret(action)
        output.append(regret)

    return agent, output
    
k = 2
n = 1000
mu = [0.1, 0.9]
sigma = [1.0, 1.0]

bandit = Gaussian(mu, sigma)
agents = [UCB1(k, n, v = 1), 
          TruncatedUCB(k, v = 1, epsilon = 1), 
          MoMUCB(k, v = 1, epsilon = 1), 
          CatoniUCB(k, n, v = 1, epsilon = 1)]

for agent in agents:
    results = [] 
    for _ in range(1): 
        agent.reset()
        agent, output = run(n, agent, bandit)
        results.append(output)
    plt.plot(range(n + 1), np.mean(results, axis = 0), label = agent.name())


plt.ylabel(r'Cumulative Regret ($R_t$)')
plt.xlabel(r'Round ($t$)')
plt.legend()
plt.savefig('example.png')
