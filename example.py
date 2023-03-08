import numpy as np
import matplotlib.pyplot as plt

from src.agents import *
from src.bandits import *

k = 2
n = 1000
mu = [0.1, 0.9]
sigma = [1.0, 1.0]

# bandit = Gaussian(mu, sigma)
bandit = Pareto(a = 2, b = 1, loc = [0.0, 0.9])
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
plt.savefig('pareto_example.png')
