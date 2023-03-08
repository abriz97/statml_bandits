import math
import random
import numpy as np

class Agent: 
    def name(self): 
        raise NotImplementedError()
        
    def reset(self): 
        raise NotImplementedError()

    def policy(self, t): 
        '''
        uses historical data to select the next action

        Input: 
            t (int): index of current round

        Output: 
            (int) index of the chosen action
        '''
        raise NotImplementedError()

    def update(self, action, reward): 
        '''
        updates any statistics using newly observed data

        Input: 
            action (int): index of action
            reward (float): corresponding reward
        '''
        raise NotImplementedError()
        
class OFU(Agent): 
    def __init__(self, k): 
        '''
        abstract base class of algorithms implementing the principle
        of optimism in the face of uncertainty principle

        Input: 
            k (int): size of the action set
        '''
        self.k = k

    def ucb(self, t, action): 
        '''
        computes the upper confidence bound for the given action

        Input: 
            t (int): index of current round
            action (int): index of the action

        Output: 
            (float): upper confidence bound for given action
        '''
        raise NotImplementedError()

    def reset(self): 
        self.count = {i: 0 for i in range(self.k)}
        self.rewards = {i: [] for i in range(self.k)}

    def policy(self, t): 
        values = [self.ucb(t, i) for i in range(self.k)]

        highest = np.max(values)
        candidates = [i for i, j in enumerate(values) if j == highest]
        return random.choice(candidates)


    def update(self, action, reward): 
        self.count[action] += 1
        self.rewards[action].append(reward)
        
class UCB1(OFU): 
    def __init__(self, k, n, v): 
        '''
        classic upper confidence bound strategy for subgaussian rewards

        Input:
            k (int): size of the action set 
            n (int): total number of decisions
            v (int): variance-proxy of rewards
        '''
        super().__init__(k)
        self.reset()
        self.v = v 

        # set the confidence level
        self.delta = pow(n, -2) 
    
    def name(self):
        return 'UCB1'

    def ucb(self, t, action): 
        n = self.count[action]
        if n < 1: 
            return np.inf

        # compute the estimated expected reward
        mu = np.mean(self.rewards[action])

        # compute the exploration bonus
        bonus = np.sqrt(2*self.v)*np.sqrt(np.log(1/self.delta)/n)

        return mu + bonus
        
class TruncatedUCB(OFU): 
    def __init__(self, k, v, epsilon): 
        '''
        upper confidence bound strategy using the truncated mean estimate 

        Input:
            k (int): size of the action set 
            v (float): bound on (1 + epsilon)-th raw moment
            epsilon (float, (0, 1]): existing raw moments of distribution 
        '''
        super().__init__(k)
        self.reset()

        self.v = v
        self.epsilon = epsilon


    def name(self):
        return 'Truncated'

    def ucb(self, t, action): 
        n = self.count[action]
        samples = self.rewards[action]
        if n < 1: 
            return np.inf
        
        # set confidence level
        delta = pow(t, -2)

        # sum the non-truncated samples
        rsum = 0.0
        for s, sample in enumerate(self.rewards[action]): 
            num = self.v * (s + 1)
            den = np.log(1/delta)
            threshold = pow(num/den, 1/(1 + self.epsilon))
            if sample > threshold: 
                continue
            else: 
                rsum += sample
        
        # compute the estimated expected reward
        mu = rsum/n

        # compute the exploration bonus
        p = self.epsilon/(1 + self.epsilon)
        c = 4*pow(self.v, 1/(1 + self.epsilon))
        bonus = c*pow(np.log(1/delta)/n, p)

        return mu + bonus
        
class MoMUCB(OFU): 
    def __init__(self, k, v, epsilon): 
        '''
        upper confidence bound strategy using median-of-means estimator

        Input:
            k (int): size of the action set 
            v (float): bound on (1 + epsilon)-th centered moment
            epsilon (float, (0, 1]): existing raw centered moments 
        '''
        super().__init__(k)
        self.reset()

        self.v = v
        self.epsilon = epsilon

    def name(self):
        return 'Median-of-Means'

    def ucb(self, t, action): 
        n = self.count[action]
        samples = self.rewards[action]
        if n < 32 * np.log(t) + 2: 
            return np.inf

        # set confidence level
        delta = pow(t, -2)

        # create batches of samples 
        m = math.floor(min(8*np.log(1/delta) + 1, n/2))
        M = math.floor(n/m)

        # compute the means 
        means = []
        for i in range(m):
            start = i * M
            finish = max((i + 1)*M, n)
            if M < n:
                mean = np.mean(samples[start: finish])
            else: 
                mean = np.mean(samples[start: ])
            means.append(mean)

        # compute the estimated expected reward
        mu = np.median(means)

        # compute the exploration bonus
        p = self.epsilon/(1 + self.epsilon)
        c = pow(12 * self.v, 1/(1 + self.epsilon))
        bonus = c*pow(2 + 16*np.log(1/delta)/n, p)

        return mu + bonus

class CatoniUCB(OFU): 
    def __init__(self, k, n, v, epsilon): 
        '''
        upper confidence bound strategy using the catoni estimator

        Input:
            k (int): size of the action set 
            n (int): total number of decisions
            v (float): bound on (1 + epsilon)-th centered moment
            epsilon (float, (0, 1]): existing centered moments
        '''
        super().__init__(k)
        self.reset()

        self.n = n
        self.v = v
        self.epsilon = epsilon

    def name(self):
        return 'Catoni'

    def f(self, muhat, alpha, rewards): 
        '''
        finding zeros of this function gives us the estimator

        Input: 
            muhat (float): guess of the estimator
            alpha (float): positive tuning parameter
            rewards (array): contains rewards observed so far

        Output: 
            (float): evaluation of the function
        '''
        x = alpha * (rewards - muhat)
        return np.sum(self.psi(x))
        
    def psi(self, x): 
        '''
        widest possible choice of influence function compatible with 
        constaints

        Input:
            x (array): values to pass through the function
        '''
        shape = x.shape[0]
        positive = np.log(1 + x + .5*x**2)
        negative = -np.log(1 - x + .5*x**2)
        return np.where(x > np.zeros(shape), positive, negative)
        
    def ucb(self, t, action, tol = 1e-02, maxit = 100): 
        n = self.count[action]
        samples = self.rewards[action]

        delta = pow(t, -2)
        if n < max(1, 4 * np.log(1/delta)): 
            return np.inf

        # compute the value of alpha
        anum = 2 * np.log(1/delta)
        aden = self.n * (self.v + self.v * anum/(self.n - anum))
        alpha = np.sqrt(anum/aden)

        # compute the mean via bisection
        low = np.min(samples) - 1.0 # guarantees f(x) > 0
        high = np.max(samples) + 1.0 # guarantees f(x) < 0

        for i in range(maxit): 
            mid = 0.5*(high + low)
            if np.abs(high - low) < tol: 
                break

            flow = self.f(low, alpha, samples)
            fmid = self.f(mid, alpha, samples)
            fhigh = self.f(high, alpha, samples)

            prod = flow * fmid
            if prod > tol: 
                flow = fmid
            else: 
                fhigh = fmid
        mu = mid

        # compute the exploration bonus
        bonus = 2*np.sqrt(self.v * np.log(1/delta)/n)

        return mu + bonus
