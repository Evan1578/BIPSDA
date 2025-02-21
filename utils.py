import torch
import numpy as np
import math
import sys
import os

sys.path.append(os.path.join( os.path.dirname(os.path.abspath(__file__)), 'externals/simplediffusion'))
from externals.simplediffusion.distributions import GaussianMixtureDistribution, GaussianDistribution

# wrapper class to make torch implementation of priors compatible with PyDream
class Parameter:
    def __init__(self, prior_dist, dim):
        self.prior_dist = prior_dist
        self.dsize = dim

    def prior(self, q0):
        return self.prior_dist.log_prob(torch.tensor(q0, dtype=torch.float32)).numpy().astype(np.float64)

    def random(self, reseed=False):
        return self.prior_dist.sample().numpy().astype(np.float64)
    
    def interval(self, alpha=1):
        """Return the interval for a given alpha value."""
        lower = [-np.inf] * self.dsize
        upper = [np.inf] * self.dsize
        return [lower, upper]
    
def mix_to_device(dist, device):
     weights = dist.weights
     distributions = []
     for distribution in dist.distributions:
          mean = distribution.mean
          cov_matrix = distribution.covariance_matrix
          new_subdist = GaussianDistribution(mean.to(device), cov_matrix.to(device))
          distributions.append(new_subdist)
     new_dist = GaussianMixtureDistribution(distributions, weights)
     return new_dist

# extract samples from Pydream chain
def extract_samples_from_chain(chains, num_samples):
    chain_length, dim = chains[0].shape
    num_chains = len(chains)
    assert num_samples <= num_chains*(chain_length // 2) 
    # remove first half of chain for burn in
    params_noburnin = np.zeros((num_chains, chain_length //2 , dim))
    for idx in range(len(chains)):
        params_noburnin[idx, :, :] = chains[idx][chain_length // 2 :, :]
    params_noburnin = params_noburnin.reshape((-1, dim))
    np.random.shuffle(params_noburnin)
    pydream_samples = torch.tensor(params_noburnin[:num_samples, :], dtype=torch.float32)
    return pydream_samples

class Likelihood:
        def __init__(self, operator, measurement, is_gaussian=True):
            self.operator = operator
            self.measurement = measurement
            self.is_gaussian = is_gaussian
        def __call__(self, params):
            likelihood_log_prob = self.operator.log_likelihood(torch.tensor(params, dtype=torch.float32), self.measurement).numpy().astype(np.float64)
            if self.is_gaussian:
                likelihood_log_prob = likelihood_log_prob - (.5 * self.operator.meas_dim * math.log(2*math.pi)) -self.operator.meas_dim * math.log(self.operator.sigma)
            return np.squeeze(likelihood_log_prob)
        
class TotalProb:
        
        def __init__(self, prior, operator, measurement):
            self.prior = prior
            self.operator = operator
            self.measurement = measurement

        def __call__(self, params):
            likelihood_log_prob_simple = self.operator.log_likelihood(torch.tensor(params, dtype=torch.float32), self.measurement).numpy().astype(np.float64)
            likelihood_log_prob = likelihood_log_prob_simple - (.5 * self.operator.meas_dim * math.log(2*math.pi)) -self.operator.meas_dim * math.log(self.operator.sigma)
            prior_log_prob = self.prior.log_prob(torch.tensor(params, dtype=torch.float32)).numpy().astype(np.float64)
            return prior_log_prob + likelihood_log_prob