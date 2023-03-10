import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt

from src.agents import *
from src.bandits import *


def true_central(a, b, n): 
	if n == 2: 
		return a*pow(b, 2)/(pow(a - 1, 2)*(a - 2))
	elif n == 3: 
		num = 2 * a * (a + 1) * pow(b, 3)
		den = pow(a - 1, 3) *(a - 2)*(a - 3)
		return num/den
	else: 
		num = 3 * a * (3* pow(a, 3) + a + 2)*pow(b, 4)
		den = pow(a - 1, 4)*(a - 2)*(a - 3)*(a - 4)
		return num/den
		
def vcentral(a, b, n): 
	vcent = pow(1 - a, a - n) * pow(-a, n - a)
	vcent = vcent * a * pow(b, n) 
	vcent = vcent * math.gamma(a - n) 
	vcent = vcent * math.gamma(n + 1)
	vcent = vcent/math.gamma(a - n + n + 1)
	
	return vcent.real

epsilon = round(float(sys.argv[1]), 2)

# parameters of pareto distribution
b = 1
a = epsilon + 1.05

# experimental settings
k = 2
n = 10000
delta = [0.0, 2.0]
bandit = Pareto(a = a, b = b, loc = delta)

# compute upper bound of (centered) moments
vraw = a/(a - (1 + epsilon))
vucb = vcentral(a = 2.01, b = 1, n = 2)
vcent = vcentral(a, b, 1 + epsilon)

print(vraw, vcent)

agents = [UCB1(k, n, v = vucb),
	      MoMUCB(k, vcent, epsilon),
		  CatoniUCB(k, n, vcent, epsilon),
		  TruncatedUCB(k, vraw, epsilon)]
		   
for agent in agents:
    results = [] 
    start = time.time()
    for _ in range(1): 
        agent.reset()
        agent, output = run(n, agent, bandit)
        results.append(output)
        
    finish = time.time()
    print(f'{agent.name()} Runtime: ', finish - start)
    plt.plot(range(n + 1), np.mean(results, axis = 0), label = agent.name())
    plt.legend()
    if epsilon < 1: 
    	plt.savefig('ParetoFM.png')
    else: 
    	plt.savefig('ParetoFV.png')

