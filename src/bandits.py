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
