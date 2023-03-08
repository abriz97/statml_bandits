import numpy as np

class Bandit: 
    def __init__(self, mu):
        '''
        Multi-Armed Bandit environment

        Input: 
            mu (list): expected reward of each action
        '''
        self.mu = mu

    def pull(self, action): 
        '''
        returns a reward from some action-specific distribution

        Input: 
            action (int): index of chosen action
        
        Output: 
            (float): random reward
        '''
        raise NotImplementedError()

    def regret(self, action): 
        '''
        compares expected reward of optimal and chosen actions

        Input: 
            action (int): index of chosen action
        
        Output: 
            (float): difference between optimal and chosen expected reward
        '''
        return max(self.mu) - self.mu[action]
        
class Gaussian(Bandit): 
    def __init__(self, mu, sigma): 
        '''
        Gaussian Multi-Armed Bandit environment

        Input: 
            mu (list): expected reward of each action
        '''
        super().__init__(mu)
        self.sigma = sigma

    def pull(self, action): 
        loc = self.mu[action]
        scale = self.sigma[action]
        return np.random.normal(loc, scale)
        
class Pareto(Bandit): 
	def __init__(self, a, b, loc): 
		'''
		Pareto Multi-Armed Bandit environment
		
		Input: 
			a (float): shape parameter 
			b (float): minimum value of rewards
			loc (list): locations of the distributions
		'''
		super().__init__([c + (a*b/(a - 1)) for c in loc])
		
		self.a = a # initialise shape parameter
		self.b = b # initialise scale parameter
		self.loc = loc # initialise locations
		
	def pull(self, action): 
		loc = self.loc[action]
		uniform = np.random.uniform()
		paretorv = self.b * np.power(1 - uniform, -1/self.a)
		return paretorv + loc

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
