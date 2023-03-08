import os
import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as studentt
from scipy.special import gamma
import numpy as np

from agents import *
from bandits import *

nits = 30
fnames = []
epsilons = []
for epsilon in [0.01, 0.20, 0.40, 0.60, 0.80, 1.0]:
	fname = str(epsilon).split('.')
	fnames += [fname[0] + fname[1]]*nits
	epsilons += [epsilon]*nits
	
def estimate_central_moment_studentt(df, power, iter_no):
    central_moment = np.mean(np.power(np.abs(studentt.rvs(df=df, size=iter_no)), power))
    return central_moment


def even_central_moment_studentt(df, power):
    return 1/(np.sqrt(np.pi) * gamma(df/2)) * (gamma((power+1)/2) * gamma((df-power)/2) * df**(power/2))

df = 5.0
for power in [1.1, 1.3, 1.5, 1.8]:
    print(estimate_central_moment_studentt(df, power, 100000) - even_central_moment_studentt(df, power))


# get arguments from command line
s = sys.argv[0].split('.')[0]
i = int(sys.argv[1])

# get the epsilon value
fname = fnames[i]
epsilon = epsilons[i]

# parameters of studentt distribution
df = epsilon + 1.05

# experimental settings
k = 2
n = 1000
delta = [0.0, 2.0]
iter_no = 25000
bandit = Studentt(df = df, loc = delta)

# compute upper bound of (centered) moments
vucb = even_central_moment_studentt(df = 2.01, power = 2)
vcent = even_central_moment_studentt(df, 1 + epsilon)

agent = MoMUCB(k, n, vcent, epsilon)
agent, output = run(n, agent, bandit)


cwd = os.getcwd()
filename = cwd + f'/out/{s}_eps{fname}_it{i%30}.npy'
np.save(filename, output)
