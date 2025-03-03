from abc import ABC
import math

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

def create_distribution(config):
    dist_name = config.data.distribution_name
    distribution = eval(dist_name)(config.data.dim)
    return distribution

def generate_cov(d, method='principled', components=None):
    # generates a random (invertible) covariance matrix of size d x d
    if method == 'principled':
        if components is None:
            components = torch.rand(d) * 2 + 10 ** (-6)
        A = torch.randn((d, d))
        Q, _ = torch.linalg.qr(A)
        Cov = Q.t() @ torch.diag(components) @ Q
    elif method == 'old':
        while True:
            A = torch.randn((d, d))
            Cov = torch.matmul(A, A.t())
            if torch.linalg.matrix_rank(Cov) == d:
                break
    return Cov

def gaussian_mixture(dim):
    distribution1 = GaussianDistribution(-5*torch.ones(dim), torch.eye(dim))
    distribution2 = GaussianDistribution(torch.zeros(dim), torch.diag(torch.linspace(1, 2, dim)))
    distribution3 = GaussianDistribution(5*torch.ones(dim), generate_cov(dim, method='principled', components=torch.linspace(1, 2, dim)))
    distributions = [distribution1, distribution2, distribution3]
    weights = [.4, .3, .3]
    distribution = GaussianMixtureDistribution(distributions, weights)
    return distribution


class GaussianDistribution(MultivariateNormal, ABC):

    # Pytorch multivariate normal distribution with added score function
    def __init__(self, mean, covariance_matrix):
        super().__init__(mean, covariance_matrix=covariance_matrix)

    def score_function(self, x):
        # input: Pytorch tensor of size batch_size x d
        # output: Pytorch tensor of size batch_size x d
        return -torch.matmul(x - self.mean[None, :], self.precision_matrix)

    def sample(self, num_samples=1):
        sample_shape = torch.Size([num_samples])
        return super().sample(sample_shape=sample_shape)

class GaussianMixtureDistribution:
    # create a distribution as weighted sum of mixture of two distributions
    def __init__(self, distributions, weights):
        self.distributions = distributions
        self.num_distributions = len(distributions)
        self.weights = weights

    def log_prob(self, x):
        exp_of_term = 0
        for idx, distribution in enumerate(self.distributions):
            exp_of_term += self.weights[idx] * torch.exp(distribution.log_prob(x))
        return torch.log(exp_of_term)

    def score_function(self, x):
        log_probs = torch.zeros(len(self.distributions), x.shape[0])
        for idx, distribution in enumerate(self.distributions):
            log_probs[idx, :] = distribution.log_prob(x)
        normalization_term = torch.max(log_probs, 0)[0]
        constant_term = 0
        for idx, distribution in enumerate(self.distributions):
            constant_term += self.weights[idx] * torch.exp(distribution.log_prob(x) - normalization_term)
        val = torch.zeros_like(x)
        for idx, distribution in enumerate(self.distributions):
            prob_correction = torch.unsqueeze(torch.exp(distribution.log_prob(x) - normalization_term)/constant_term, 1)
            val += self.weights[idx] * prob_correction * distribution.score_function(x)
        return val

    def score_function_2(self, x):
        term1 = 1 / (torch.max(torch.exp(self.log_prob(x)), torch.tensor([10 ** (-25)])))
        other_terms = 0
        for idx, distribution in enumerate(self.distributions):
            other_terms += self.weights[idx] * torch.unsqueeze(torch.exp(distribution.log_prob(x)),
                                                               dim=1) * distribution.score_function(x)
        return torch.unsqueeze(term1, dim=1) * other_terms

    def sample(self, num_samples=1):
        which_dist = torch.tensor(
            np.random.choice(self.num_distributions, size=(num_samples, 1), replace=True, p=self.weights))
        samples = []
        for idx, distribution in enumerate(self.distributions):
            samples.append((which_dist == idx) * distribution.sample(num_samples=num_samples))
        cross = torch.stack(samples)
        return cross.sum(dim=0)
    
def get_convolved_distribution(distribution, noise_level):
    # returns the distribution obtained by convolving a distribution with a Gaussian at a fixed noise level
    if isinstance(distribution, GaussianDistribution):
        dim = len(distribution.mean)
        convolved_distribution = GaussianDistribution(distribution.mean, distribution.covariance_matrix + noise_level ** 2 * torch.eye(dim))
    elif isinstance(distribution, GaussianMixtureDistribution):
        convolved_dists = []
        for gauss_distribution in distribution.distributions:
            covariance = gauss_distribution.covariance_matrix
            mean = gauss_distribution.mean
            dim = len(mean)
            dist_convolved = GaussianDistribution(mean, covariance + noise_level ** 2 * torch.eye(dim))
            convolved_dists.append(dist_convolved)
        convolved_distribution = GaussianMixtureDistribution(convolved_dists, distribution.weights)

    return convolved_distribution


def get_mixture_gaussian_posterior(dist, y, A, sigma):
    """
    Returns the Gaussian mixture posterior distribution obtained when the prior is a Gaussian mixture
    and the likelihood is a linear Gaussian model with white noise, i.e., y = Ax + z, z \sim N(0, \sigma^2 I). 
    Inputs:
        dist (instance of GaussianMixtureDistribution): the prior distribution
        y (1D torch.Tensor of length D2): the measurements
        A (2D torch.Tensor size D1 x D2): the forward operator
        sigma (positive scalar): the standard deviation of the noise in the forward model
    Outputs:
        The posterior Gaussian mixture (instance of GaussianMixtureDistribution)
    """
    distributions = dist.distributions
    num_distributions = len(distributions)
    A_tA = A.t() @ A
    weights = dist.weights
    new_distributions = []
    new_weights = []
    for idx in range(num_distributions):
        # get weight and distribution of ith component of mixture
        weight = weights[idx]
        prior_gaussian = distributions[idx]
        # compute the updated mean and covariance
        new_precision_matrix = prior_gaussian.precision_matrix + (1/(sigma*sigma))*A_tA
        new_covariance_matrix = torch.linalg.inv(new_precision_matrix)
        term_1 = prior_gaussian.precision_matrix @ prior_gaussian.mean + (1/(sigma*sigma))*(A.t() @ y)
        new_mean = new_covariance_matrix @ term_1
        posterior_distribution = GaussianDistribution(new_mean, new_covariance_matrix)
        # compute the new (unnormalized) weight
        d_i = torch.exp(-.5 * ((y.t()@y)*(1/(sigma*sigma)) - new_mean.t() @ new_precision_matrix @ new_mean + prior_gaussian.mean.t() @ prior_gaussian.precision_matrix @ prior_gaussian.mean))
        det_prior_inv = 1/ (torch.linalg.det(prior_gaussian.covariance_matrix))**(1/2)
        det_posterior = torch.linalg.det(new_covariance_matrix)**(1/2)
        new_weight = d_i * det_prior_inv * det_posterior * weight
        new_distributions.append(posterior_distribution)
        new_weights.append(new_weight.item())
    # normalize the weights
    sum_weights = sum(new_weights)
    new_weights_normed = [new_weight/sum_weights for new_weight in new_weights]
    # return posterior distribution
    return GaussianMixtureDistribution(new_distributions, new_weights_normed)