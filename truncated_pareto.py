import os
import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt

from src.agents import *
from src.bandits import *

nits = 30
fnames = []
epsilons = []
for epsilon in [0.01, 0.20, 0.40, 0.60, 0.80, 1.0]: 
	fname = str(epsilon).split('.')
	fnames += [fname[0] + fname[1]]*nits
	epsilons += [epsilon]*nits
	
	
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


# get arguments from command line
s = sys.argv[0].split('.')[0]
i = int(sys.argv[1])

# get the epsilon value
fname = fnames[i]
epsilon = epsilons[i]

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

agent = TruncatedUCB(k, vraw, epsilon)
agent, output = run(n, agent, bandit)

cwd = os.getcwd()
filename = cwd + f'/out/{s}_eps{fname}_it{i%30}.npy'
np.save(filename, output)
