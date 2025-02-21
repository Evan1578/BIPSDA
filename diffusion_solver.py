import torch.nn as nn
import tqdm
from externals.daps.sampler import Scheduler as DiffusionScheduler, DiffusionSampler, Trajectory
import torch
import math
import numpy as np
import time
from abc import ABC, abstractmethod
import logging
from torch.autograd.functional import vjp, jacobian, hessian
from utils import Parameter, Likelihood, extract_samples_from_chain
from pydream import core
from pydream import convergence
import sys
import random
import os
sys.path.append(os.path.join( os.path.dirname(os.path.abspath(__file__)), 'externals/simplediffusion'))
from externals.simplediffusion import distributions

'''
The main class here is GIPSDA, which will be a generalized version of the DAPS class in externals/daps/sampler.py. It will be generalized in the 
sense that:
(1) The method can be used to either compute samples from the posterior (as in DAPS) or estimate the MAP point. These two options correspond to two 
different ways of incorporating measurement information from the inverse problem: using Langevin dynamics (as in DAPS), or using a proximal 
problem solve, respectively.  
(2) The forward diffusion process incorporates a parameter that can be used to toggle between a "fully stochastic" and "fully deterministic" diffusion
process. 
'''

def get_solver(**kwargs):
    latent = kwargs['latent']
    kwargs.pop('latent')
    if latent:
        raise NotImplementedError
    return GIPSDA(**kwargs)


class MeasurementSolver(ABC):
    """"
    Abstract class for the prediction distribution consitency step in BIPSDA 
    """
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def solve(self, x0hat, operator, measurement, cinv_op, ratio, cov_type, record=False, verbose=False):
        """
        x0hat (torch.tensor)
        operator (instance of subclass of daps.forward_operator.Operator)
        measurement (torch.tensor)
        cinv_op (implementation of inverse of covariance operator for p(m(0) | m(t)))
        ratio (positive scalar)
        record (bool)
        """
        pass



class mMALASampler(MeasurementSolver):

    def __init__(self, step_size, num_its, metric_type = 'none'):
        super().__init__()
        self.step_size = step_size
        self.num_its = num_its
        self.metric_type = metric_type
        self.failed_indices = []

    def solve(self, x0hat, operator, measurement, cinv_op, ratio, cov_type, record=False, verbose=False):
        # check magnitude of x0hat
        if torch.max(torch.abs(x0hat)) > 200:
            vals, _ = torch.max(torch.abs(x0hat), dim=1)
            escape_check = vals > 200
            num_escape = torch.count_nonzero(escape_check).item()
            escape_indices = torch.nonzero(escape_check)
            logging.info("Oh no! {:d} samples escaped [-200, 200]^N Cube. Resetting these samples to the origin and marking as failed".format(num_escape))
            for idx in range(escape_indices.shape[0]):
                val = escape_indices[idx, :].item()
                if val not in self.failed_indices:
                    self.failed_indices.append(val)
            with torch.no_grad():
                x0hat = x0hat * (~ escape_check[:, None])
        # form prior precision matrix
        if cov_type == 'identity':
            precision = cinv_op*torch.unsqueeze(torch.eye(x0hat.shape[1], device=x0hat.device), dim=0).expand(x0hat.shape[0], -1, -1)
        elif cov_type == 'exact':
            precision = cinv_op
        # form metric tensor, inverse, and square root
        metric_tens = self.get_metric_tensor(x0hat, precision, operator, measurement)
        #eig_vals = torch.linalg.eigvals(metric_tens)
        #logging.info("The minimum eigenvalue (real part) was {:.4f}".format(torch.min(torch.real(eig_vals))))
        #logging.info("The maximum absolute value of the eigenvalues (imaginary part) was {:.4f}".format(torch.max(torch.abs(torch.imag(eig_vals)))))
        try:
            metric_tens = self.get_metric_tensor(x0hat, precision, operator, measurement)
            sqrt_metric_tens = torch.linalg.cholesky(metric_tens)
        except:
            logging.info("Cholesky decomp of Gauss-Newton hessian for at least one of the samples failed - defaulting to precision matrix for preconditioning")
            metric_tens = precision
            sqrt_metric_tens = torch.linalg.cholesky(metric_tens)
        inv_metric_tens = torch.linalg.inv(metric_tens)
        sqrt_inv_metric_tens = torch.linalg.inv(sqrt_metric_tens)
        # set parameters and initial point
        x = x0hat.clone().detach().requires_grad_(True)
        pbar = tqdm.trange(self.num_its) if verbose else range(self.num_its)
        step_size = self.step_size
        # define log probabilities of prior and target distribution (prior + likelihood)
        def compute_prior_logprob(x): # return unnormalized log probability
            diff = x - x0hat
            term1 = torch.matmul(precision, diff.unsqueeze(-1)).squeeze(-1)
            return (-1./2.) * torch.sum(diff * term1, dim=-1)
        def compute_target_logprob(x):
            prior_logprob = compute_prior_logprob(x)
            likelihood_logprob = operator.log_likelihood(x, measurement)
            return prior_logprob + likelihood_logprob
        # compute mean of proposal distribution from  initialization
        loss = torch.sum(compute_target_logprob(x))
        loss.backward()
        grad = x.grad.data.detach() # form gradient of log probability
        with torch.no_grad():
            mean = x.data + ((step_size**2)/2) * torch.squeeze(torch.matmul(inv_metric_tens, torch.unsqueeze(grad, dim=-1)))
        # begin iterations
        trac_acceptances = np.zeros(self.num_its)
        for idx in pbar:
            # get sample from proposal distribution
            with torch.no_grad():
                white_noise = torch.randn_like(mean)
                noise_sample = step_size * torch.squeeze(torch.matmul(sqrt_inv_metric_tens, torch.unsqueeze(white_noise, dim=-1)))
                prop = mean + noise_sample
            # get mean of proposal distribution corresponding to proposed point
            prop.requires_grad_(True)
            loss = torch.sum(compute_target_logprob(prop))
            loss.backward()
            prop_grad = prop.grad.data.detach()
            with torch.no_grad():
                prop_mean = prop.data + ((step_size**2)/2) * torch.squeeze(torch.matmul(inv_metric_tens, torch.unsqueeze(prop_grad, dim=-1)))
            # compute acceptance probability and and make acceptance decision
            with torch.no_grad():
                accept_prob = self.get_accept_prob(x, mean, prop, prop_mean, compute_target_logprob, metric_tens / (step_size * step_size))
            rand_sample = torch.rand(accept_prob.shape[0], device=x0hat.device)
            accept = rand_sample < accept_prob
            # update iterate and proposal mean
            x.data = (x.data)*torch.logical_not(accept)[:, None] + (prop)*(accept[:, None])
            mean = (mean)*torch.logical_not(accept)[:, None] + (prop_mean)*(accept[:, None])
            # compute summary statistic
            trac_acceptances[idx] = torch.count_nonzero(accept)/ len(accept)
            if verbose:
                pbar.set_postfix({'Acc. Ratio': '{:.4f}'.format(trac_acceptances[idx])})
        return x.detach()

    def get_metric_tensor(self, x0hat, precision, operator, measurement):
        if self.metric_type == 'none':
            metric_tens =torch.unsqueeze(torch.eye(x0hat.shape[1], device=x0hat.device), dim=0).expand(x0hat.shape[0], -1, -1)
        elif self.metric_type == 'simple':
            term1 = operator.At @ operator.A / (operator.sigma * operator.sigma)
            metric_tens = term1.to(x0hat.device)[None, :, :] + precision
        elif self.metric_type == 'gauss_newton':
            if operator.name == 'gaussphaseretrieval1D':
                Ax = torch.squeeze(torch.matmul(torch.unsqueeze(operator.A.to(x0hat.device), dim=0), torch.unsqueeze(x0hat, dim=-1)))
                Ax_diag = torch.diag_embed(Ax)
                Jacob_fwd_model = 2 * torch.matmul(torch.unsqueeze(operator.At.to(x0hat.device), dim=0), Ax_diag)
                batch_gauss_newton_hessian = (1 / (operator.sigma * operator.sigma)) * torch.matmul(Jacob_fwd_model, torch.transpose(Jacob_fwd_model, 1, 2))
                avg_gauss_newton_hessian = torch.mean(batch_gauss_newton_hessian, dim=0, keepdim=True)
            else:
                raise Exception("Gauss Newton Hessian approximation not coded for this operator type")
            return avg_gauss_newton_hessian + precision
        elif self.metric_type == 'exact':
            if operator.name == 'xray_tomography':
                I0 = operator.I0
                def diag_batch(tensor):
                    return torch.diag(tensor)
                diag_term = torch.vmap(diag_batch)(operator(x0hat))
                first_part = torch.matmul(diag_term, torch.unsqueeze(operator.A, dim=0))
                batch_hessian = torch.matmul(torch.unsqueeze(operator.At, dim=0), first_part)
                avg_hessian = torch.mean(batch_hessian, dim=0, keepdim=True)
                return avg_hessian + precision
            else:
                raise Exception("Exact Hessian computation not coded for this operator type")
            raise Exception("Method not coded yet!")
        return metric_tens

    def get_accept_prob(self, current, cur_mean, proposed, prop_mean, compute_target_logprob, trans_prec):
        fwd_diff = torch.unsqueeze(proposed - cur_mean, dim=-1)
        transition_logprob = (-1./2.)* torch.sum(torch.squeeze(fwd_diff) * torch.squeeze(torch.matmul(trans_prec, fwd_diff)), dim=-1)
        reverse_diff = torch.unsqueeze(current - prop_mean, dim=-1)
        reverse_logprob = (-1./2.)* torch.sum(torch.squeeze(reverse_diff) * torch.squeeze(torch.matmul(trans_prec, reverse_diff)), dim=-1)
        log_acceptance = compute_target_logprob(proposed) - compute_target_logprob(current) + reverse_logprob - transition_logprob
        acceptance = torch.exp(log_acceptance)
        return torch.clip(acceptance, min=0, max=1)
        



class PyDreamSampler(MeasurementSolver):
    """
    Measurement Solver that uses PyDream to sample from the prediction distribution. 
    Requires for loop over the batch dimension. 
    """

    def __init__(self, num_chains, pydream_its):
        self.num_chains = num_chains
        self.pydream_its = pydream_its 
        super().__init__()

    def solve(self, x0hat, operator, measurement, cinv_op, ratio, cov_type, record=False, verbose=False):
        # form covariance matrix 
        if cov_type == 'identity':
            precision = cinv_op*torch.unsqueeze(torch.eye(x0hat.shape[1], device=x0hat.device), dim=0).expand(x0hat.shape[0], -1, -1)
        elif cov_type == 'exact':
            precision = cinv_op
        cov_matrix = torch.linalg.inv(precision)
        x = torch.zeros_like(x0hat)
        pbar = tqdm.trange(x0hat.shape[0]) if verbose else range(x0hat.shape[0])
        init_time = time.time()
        for idx in pbar:
            # form prior 
            prior = distributions.GaussianDistribution(x0hat[idx, :].to("cpu"), cov_matrix[idx, :, :].to("cpu"))
            # form Pydream-compatible likelihood and prior
            parameter = Parameter(prior, operator.data_dim)
            likelihood = Likelihood(operator, torch.reshape(measurement[idx, :].to("cpu"), (1, -1)))
            # run pyDream
            nchains = self.num_chains
            pydream_its = self.pydream_its
            params, _ = core.run_dream([parameter], likelihood, nchains=nchains, niterations=pydream_its, save_history=False, verbose=False, start_random=True)
            # compute GR summary statistic
            GR = convergence.Gelman_Rubin(params)
            x[idx, :] = torch.tensor(params[0][-1, :], dtype=torch.float32, device=x.device)
            if verbose:
                pbar.set_postfix({'Max GR Val': '{:.4f}'.format(np.max(GR))})
        if verbose:
            elapsed_time = time.time() - init_time
            logging.info("Tbe sampling took {:.4f} seconds to complete {:d} iterations".format(elapsed_time, x0hat.shape[0]))
        return x
        


class ExactSampler(MeasurementSolver):
    """
    Measurement solver for the case where the prediction distribution is a Gaussian, which holds when the likelihood is linear-Gaussian
    """

    def __init__(self):
        super().__init__()


    def solve(self, x0hat, operator, measurement, cinv_op, ratio, cov_type, record=False, verbose=False):

        # form posterior covariance
        term1 = torch.unsqueeze(operator.At @ operator.A / (operator.sigma * operator.sigma), dim=0).to(x0hat.device) # 1 x n x n
        if cov_type == 'identity':
            term2 = cinv_op*torch.unsqueeze(torch.eye(x0hat.shape[1], device=x0hat.device), dim=0).expand(x0hat.shape[0], -1, -1)
        elif cov_type == 'exact':
            term2 = cinv_op
        post_prec = term1 + term2
        post_cov = torch.linalg.inv(post_prec)
        post_cov = (torch.transpose(post_cov, 1, 2) + post_cov) / 2 # symmetrize 

        # form posterior mean
        post_mean = torch.squeeze(torch.matmul(post_cov, torch.matmul(term2, torch.unsqueeze(x0hat, dim=-1)) + torch.matmul(measurement, operator.A.to(measurement.device))[:, :, None] /(operator.sigma * operator.sigma) ))

        # compute samples 
        dist = torch.distributions.multivariate_normal.MultivariateNormal(post_mean, covariance_matrix=post_cov)
        samples = dist.sample()

        return samples


class LangevinDynamics(MeasurementSolver):
    """
    Generalized version of Langevin dynamics class in DAPS that allows for general covariance matrics in p(m(0) | m(t)) (not just white noise covariance)
    """
    def __init__(self,  num_steps, lr, tau=0.01, lr_min_ratio=0.01):
        super().__init__()
        self.num_steps = num_steps
        self.lr = lr
        self.tau = tau
        self.lr_min_ratio = lr_min_ratio
    
    def solve(self, x0hat, operator, measurement, cinv, ratio, cov_type, record=False, verbose=False):
        #TODO: generalize so can work with multiple x0hats? 
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        lr = self.get_lr(ratio)
        x = x0hat.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([x], lr)
        cinv_op = self._get_cov_operator(cinv, cov_type)
        for _ in pbar:
            optimizer.zero_grad()
            #operator.error(x, measurement).sum() / (2 * self.tau ** 2)
            loss = -1. * torch.sum(operator.log_likelihood(x, measurement))
            loss += (1/2)* torch.sum((x - x0hat)*(cinv_op(x-x0hat)))
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                epsilon = torch.randn_like(x)
                x.data = x.data + np.sqrt(2 * lr) * epsilon
            # early stopping with NaN
            if torch.isnan(x).any():
                return torch.zeros_like(x)
            # hack for inf values
            if torch.isinf(x).sum() > 0:
                print("Infinite Values!")
                x = x * torch.bitwise_not(torch.isinf(x))
            # record
            if record:
                self._record(x, epsilon, loss)
        return x.detach()
    
    def _get_cov_operator(self, cinv, cov_type):
        if cov_type == 'identity':
            return lambda x: cinv*x
        elif cov_type == 'exact':
            return lambda x: torch.squeeze(torch.matmul(cinv, torch.unsqueeze(x, -1)))
        else:
            raise Exception("Unknown Covariance Type!")
        
        
    def get_lr(self, ratio):
        """
            Computes the learning rate based on the given ratio.
        """
        p = 1
        multiplier = (1 ** (1 / p) + ratio * (self.lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
        return multiplier * self.lr
    def _record(self, x, epsilon, loss):
        """
            Records the intermediate states during sampling.
        """
        self.trajectory.add_tensor(f'xi', x)
        self.trajectory.add_tensor(f'epsilon', epsilon)
        self.trajectory.add_value(f'loss', loss)



class CheatRTO(MeasurementSolver):
    """
    Produces an approximate sample from the posterior distribution by solving the deterministic
    proximal problem \argmin_x \frac{1}{2\sigma_n^2} ||y - A(x) + \epsilon ||_2^2 + \frac{\rho}{2} || x - \hat{x} + \tau ||_2^2, 
    where:
        - sigma_n is the standard deviation of the additive white Gaussian noise in the forward model
        - A is the forward operator
        - rho and hat{x} are provided as inputs
        -\epsilon \sim N(0, \sigma_n^2 I )
        - \tau \sim 
    """
    def __init__(self, measurement_config):
        super().__init__()
        self.proximal_solver = ProximalSolver(measurement_config)

    def solve(self, x0hat, operator, measurement, cinv, ratio, cov_type, record=False, verbose=False):
        # get measurement perturbation
        if operator.name == 'xray_tomography':
            measurement_perturb = torch.poisson(measurement) - measurement
        else:
            measurement_perturb = operator.sigma * torch.randn_like(measurement)
        # get x0hat perturbation
        if cov_type == 'identity':
            x0hat_perturb = (1/math.sqrt(cinv)) * torch.randn_like(x0hat)
        elif cov_type == 'exact':
            cov_matrix = torch.linalg.inv(cinv)
            sqrt_cov_matrix = torch.linalg.cholesky(cov_matrix)
            x0hat_perturb = torch.squeeze(torch.matmul(sqrt_cov_matrix, torch.unsqueeze(torch.randn_like(x0hat), dim=-1)))
        else:
            raise Exception("Unknown covariance type")
        # add perturbations
        perturbed_measurements = measurement + measurement_perturb
        perturbed_mean = x0hat + x0hat_perturb
        # solve proximal optimization problem 
        sol = self.proximal_solver.solve(perturbed_mean, operator, perturbed_measurements, cinv, ratio, cov_type, record=False, verbose=verbose)
        self.failed_indices = self.proximal_solver.failed_indices
        return sol


class ProximalSolver(MeasurementSolver):
    """
    Solves the proximal problem \argmin_x \frac{1}{2\sigma_n^2} ||y - A(x) ||_2^2 + \frac{\rho}{2} || x - \hat{x} ||_2^2, 
    where:
        - sigma_n is the standard deviation of the additive white Gaussian noise in the forward model
        - A is the forward operator
        - rho and hat{x} are provided as inputs
    """

    def __init__(self, measurement_config):
        """
        Measurement_config (ml_collections.ConfigDict object): Dictionary config file containing parameters relevant to proximal problem solve
        """
        super().__init__()
        self.measurement_config = measurement_config
        try:
            self.lambda_ = measurement_config['lambda_']
        except:
            self.lambda_ = 1
        self.generic_solver_params_set_flag = False
        self.failed_indices = []

    def solve(self, x0hat, operator, measurement, cinv, ratio, cov_type, record=False, verbose=False):
        if hasattr(operator, 'solve_proximal') and operator.name == 'inpainting1D':
            #raise Exception("Still need to update proximal solvers for general covariance matrices")
            sol = operator.solve_proximal(x0hat, measurement, cinv, cov_type, record=record, verbose=verbose, params=self.measurement_config)
        else: 
            if not self.generic_solver_params_set_flag:
                self.set_generic_solver_params()
            sol = self.generic_proximal_solve(operator, x0hat, measurement, cinv, cov_type, record=record, verbose=verbose)
        return sol
    
    def _get_cov_operator(self, cinv, cov_type):
        if cov_type == 'identity':
            return lambda x: cinv*x
        elif cov_type == 'exact':
            return lambda x: torch.squeeze(torch.matmul(cinv, torch.unsqueeze(x, -1)))
        else:
            raise Exception("Unknown Covariance Type!")

    
    def set_generic_solver_params(self):
        # sets parameters for generic gradient-based solver from self.measurement_config parameters 
        try:
            self.max_its = self.measurement_config['max_its']
        except:
            self.max_its = 100
        try:
            self.grad_tol = self.measurement_config['grad_tol']
        except:
            self.grad_tol = 1e-4
        try:
            self.solver = self.measurement_config['solver']
            self.solver_params = self.measurement_config['solver_params']
        except:
            self.solver = 'sgd'
            self.solver_params = {'lr': 1e-4}
        self.generic_solver_params_set_flag = True

    def _check_escape(self, point):
        if (torch.max(torch.abs(point)) > 1000) or torch.isnan(point).any():
            vals, _ = torch.max(torch.abs(point), dim=1)
            vals2, _ = torch.max(torch.isnan(point), dim=1)
            escape_check = torch.logical_or((vals > 1000), vals2)
            num_escape = torch.count_nonzero(escape_check).item()
            escape_indices = torch.nonzero(escape_check)
            logging.info("Oh no! {:d} samples escaped. Resetting these samples to the origin and marking as failed".format(num_escape))
            for idx in range(escape_indices.shape[0]):
                val = escape_indices[idx, :].item()
                if val not in self.failed_indices:
                    self.failed_indices.append(val)
            with torch.no_grad():
                point = point * (~ escape_check[:, None]) 
        return point


    def generic_proximal_solve(self, operator, x0hat, measurement, cinv, cov_type, record=False, verbose=False):
        # early stopping with NaN or if relative gradient norm less than tolerance for all samples
        x0hat = self._check_escape(x0hat)
        #verbose=True
        x = x0hat.clone().detach().requires_grad_(True)
        baseline_norm = torch.linalg.norm(torch.flatten(x0hat, start_dim=1), dim=1)
        cinv_op = self._get_cov_operator(cinv, cov_type)
        if self.solver == 'gsd' or self.solver == 'lbfgs':
            if self.solver == 'sgd':
                optimizer = torch.optim.SGD([x], **self.solver_params)
            elif self.solver == 'lbfgs':
                optimizer = torch.optim.LBFGS([x], **self.solver_params)
                def closure():
                    optimizer.zero_grad()
                    loss =  -1. * operator.log_likelihood(x, measurement).sum()   # - log_likelihood 
                    loss += (1/2)* torch.sum((x - x0hat) * cinv_op(x-x0hat))
                    loss.backward()
                    return loss
        elif self.solver == 'gauss-newton':
            optimizer = torch.optim.SGD([x]) # just for computing gradients
            if cov_type == 'identity':
                precision = cinv*torch.unsqueeze(torch.eye(x0hat.shape[1], device=x0hat.device), dim=0).expand(x0hat.shape[0], -1, -1)
            elif cov_type == 'exact':
                precision = cinv
            else: 
                raise Exception("Unknown covariance type!")
            if operator.name != 'gaussphaseretrieval1D':
                raise Exception("This method not implemented for this problem yet!")
        else: 
            raise Exception("Unknown solver type!")
        pbar = tqdm.trange(self.max_its) if verbose else range(self.max_its)
        #print("Norm of x is {:.4f}".format(torch.linalg.norm(x).item()))
        for _ in pbar:
            if self.solver == 'sgd':
                optimizer.zero_grad()
                loss =  -1. * operator.log_likelihood(x, measurement).sum()   # - log_likelihood 
                loss += (1/2)* torch.sum((x - x0hat) * cinv_op(x-x0hat))
                loss.backward()
                optimizer.step()
            elif self.solver == 'lbfgs':
                optimizer.step(closure)
            elif self.solver == 'gauss-newton':
                optimizer.zero_grad()
                def loss_fn(x):
                    loss =  -1. * operator.log_likelihood(x, measurement)   # - log_likelihood 
                    loss += torch.sum((1/2)* (x - x0hat) * cinv_op(x-x0hat), dim=-1)
                    return loss
                loss = torch.sum(loss_fn(x))
                loss.backward()
                with torch.no_grad():
                    xgrad = x.grad.detach()
                    gauss_newton_hessian = operator.form_gauss_newton_hessian(precision, x)
                    update = torch.linalg.solve(gauss_newton_hessian, xgrad)
                    x.data = self._backtracking_line_search(x.data, -1. * update.data, xgrad.data, loss_fn)
            if x.grad is not None:
                grad_norm = torch.linalg.norm(torch.flatten(x.grad, start_dim=1), dim=1)
                norm_grad_norm = torch.divide(grad_norm, baseline_norm)
                if verbose:
                    pbar.set_postfix({'Grad Norm': '{:.4f}'.format(torch.mean(norm_grad_norm).item())})
                if torch.allclose(norm_grad_norm, torch.zeros_like(grad_norm), atol=self.grad_tol):
                    if verbose:
                        logging.info("Terminating early - gradient norm less than tolerance for entire batch")
                    return x.detach()
        #logging.info("The relative gradient norm is {:.4f}".format(torch.mean(norm_grad_norm).item()))

         # early stopping with NaN or if relative gradient norm less than tolerance for all samples
        x = self._check_escape(x)
        return x.detach()
    
    def _backtracking_line_search(self, init, direc, grad, loss_fn, c=1e-4):
        inner_product = torch.sum(direc*grad, dim=-1) # inner product of gradient and search direction
        alphas = torch.zeros(init.shape[0], device=init.device) # step_sizes 
        satisfied = torch.zeros(init.shape[0], dtype=torch.bool, device=init.device)
        init_val = loss_fn(init)
        not_converge = True
        alpha_cur = 1
        alpha_min = 1e-6
        while not_converge:
            check = loss_fn(init + alpha_cur * direc) 
            is_satisfied = torch.le(check, init_val + c*alpha_cur*inner_product)
            first_satisfied = torch.logical_and(is_satisfied, torch.logical_not(satisfied))
            alphas = first_satisfied * alpha_cur
            satisfied = torch.logical_or(is_satisfied, satisfied)
            if torch.count_nonzero(satisfied) == init.shape[0]:
                break
            elif alpha_cur <= alpha_min:
                alphas = alpha_min * torch.logical_not(satisfied)
                break
            else:
                alpha_cur /= 2
        return init + torch.unsqueeze(alphas, dim=-1)*direc

    
class GeneralizedDiffusionSampler(DiffusionSampler):

    def __init__(self, scheduler, solver='euler'):
        super().__init__(scheduler, solver=solver)
    
    def sample(self, model, x_start, SDE=False, record=False, verbose=False):
        """
            Samples from the diffusion process using the specified model.

            Parameters:
                model (DiffusionModel): Diffusion model supports 'score' and 'tweedie'
                x_start (torch.Tensor): Initial state.
                SDE (bool): Whether to use Stochastic Differential Equations.
                record (bool): Whether to record the trajectory.
                verbose (bool): Whether to display progress bar.

            Returns:
                torch.Tensor: The final sampled state.
        """
        if self.solver == 'euler':
            return self._euler(model, x_start, SDE, record, verbose)
        elif self.solver == 'recursive_tweedie':
            return self._recursivetweedie(model, x_start, record, verbose)
        else:
            raise NotImplementedError
    
    def  _recursivetweedie(self, model, x_start,  record=False, verbose=False):      
        """
            Recursive tweedie method for sampling from the diffusion process.
        """
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.trange(self.scheduler.num_steps) if verbose else range(self.scheduler.num_steps)

        x = x_start
        for step in pbar:
            sigma, sigma_next = self.scheduler.sigma_steps[step], self.scheduler.sigma_steps[step + 1]
            factor = 2.* (sigma**2 - sigma_next**2) #Constant '2' is here so 'factor' has consistent interpretation across samplers
            score = model.score(x, sigma)
            x = x + factor * .5 * score  # .5 is here so 'factor' has consistent interpretation across samplers
            # record
            if record:
                self._record(x, score, sigma, factor)
        return x



class GIPSDA(nn.Module):
    """
    Implementation of GIPSDA framework for solving inverse problems. 
    """

    def __init__(self, annealing_scheduler_config, diffusion_scheduler_config, beta_scheduler_config, measurement_config, mode, cov_type, pred_alg):
        super().__init__()
        annealing_scheduler_config, diffusion_scheduler_config = self._check(annealing_scheduler_config, diffusion_scheduler_config)
        self.annealing_scheduler = annealing_scheduler = DiffusionScheduler(**annealing_scheduler_config)
        self.beta_schedule = self._get_beta_schedule(beta_scheduler_config, annealing_scheduler)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        if mode == 'langevin_dynamics':
            self.data_consistency = LangevinDynamics(**measurement_config)
        elif mode == 'map_estimation':
            self.data_consistency = ProximalSolver(measurement_config)
        elif mode == 'exact_posterior_sampling':
            self.data_consistency = ExactSampler(**measurement_config)
        elif mode == 'pydream_sampling':
            self.data_consistency = PyDreamSampler(**measurement_config)
        elif mode == 'mMALA':
            self.data_consistency = mMALASampler(**measurement_config)
        elif mode == 'cheatRTO':
            self.data_consistency = CheatRTO(measurement_config)
        else:
            raise Exception("Unknown data consistency mode provided")
        self.cov_type = cov_type
        self.pred_alg = pred_alg
        self.failed_indices = []

    def solve(self, model, x_start, operator, measurement, evaluator, record=False, verbose=False, gt=None):
        #TODO: add prediction_dist = 'Gaussian' or 'Gaussian Mixture'?
        '''
        Solve Inverse Problem in GIPSDA framework

        Parameters:
            model (nn.Module): Diffusion model
            x_start (torch.Tensor): Initial state
            operator (nn.Module): forward operator module
            measurement (torch.Tensor): inverse problem measurements
            evaluator (Evaluator): Evaluation function
        '''

        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.trange(self.annealing_scheduler.num_steps) if verbose else range(self.annealing_scheduler.num_steps)
        xt = x_start
        for step in pbar:

            #print("Norm is {:.4f}".format(torch.linalg.norm(xt).item()))

            sigma = self.annealing_scheduler.sigma_steps[step]

            # 1. Reverse diffusion:
            diffusion_scheduler = DiffusionScheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
            sampler = GeneralizedDiffusionSampler(diffusion_scheduler, solver=self.pred_alg)
            x0hat = sampler.sample(model, xt, SDE=False, verbose=False)  

            # 2. Incorporate measurement information
            cinv_op = self._get_cinv(xt, model, sigma)
            x0y = self.data_consistency.solve(x0hat, operator, measurement, cinv_op, step/self.annealing_scheduler.num_steps, self.cov_type, verbose=verbose)

            # 3. Forward Diffusion
            sigma_next = self.annealing_scheduler.sigma_steps[step + 1]
            beta = self.beta_schedule[step]
            it_change = xt - x0y
            xt = x0y + (math.sqrt(sigma_next*sigma_next - beta*beta)/sigma) * it_change  + beta * torch.randn_like(x0y)

            # 4. Evaluation
            x0hat_results = x0y_results = None
            if evaluator and gt is not None:
                with torch.no_grad():
                    x0hat_results = evaluator(gt, measurement, x0hat)
                    x0y_results = evaluator(gt, measurement, x0y)

                # record
                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        'x0hat' + '_' + main_eval_fn_name: f"{x0hat_results[main_eval_fn_name].item():.2f}",
                        'x0y' + '_' + main_eval_fn_name: f"{x0y_results[main_eval_fn_name].item():.2f}",
                    })
            if record:
                self._record(xt, x0y, x0hat, sigma, beta, x0hat_results, x0y_results)
            try:
                self.failed_indices = self.data_consistency.failed_indices
            except:
                pass
                
        return xt

    def _get_beta_schedule(self, beta_scheduler_config, annealing_scheduler):

        mode = beta_scheduler_config['mode']
        sigma_schedule_omitfirst = annealing_scheduler.sigma_steps[1:]
        
        if mode == 'deterministic':
            beta_schedule = np.zeros_like(sigma_schedule_omitfirst)
        elif mode == 'full-stoch':
            beta_schedule = sigma_schedule_omitfirst
        elif mode == 'one-half':
            beta_schedule = math.sqrt(.5)*sigma_schedule_omitfirst
        elif mode == 'squared':
            c = 1./ sigma_schedule_omitfirst[0]
            beta_schedule = c * sigma_schedule_omitfirst**2
        else:
            raise Exception("Unknown mode")

        return beta_schedule

    def _get_cinv(self, xt, model, sigma):
        if self.cov_type == 'identity':
            return 1/(sigma*sigma)
        elif self.cov_type == 'exact':
            
            # compute inverse covariance matrix 
            #logging.info("Norm of 4668th element is {:.4f}".format(torch.linalg.norm(xt[4668, :]).item()))
            fun = lambda x : torch.sum(model.score_wgrad(x, sigma), dim=0)
            init_time = time.time() 
            xt.requires_grad = True
            J = jacobian(fun, xt, create_graph=False, strict=True).detach() #J will be of size x.shape x batch x x.shape
            xt.requires_grad = False
            elapsed_time = time.time() - init_time
            #logging.info("Jacobian computation took {:.4f} seconds".format(elapsed_time))
            init_time = time.time() 
            J = torch.movedim(J, len(xt.shape) - 1, 0) #of size batch x x.shape x x.shape
            tot_size = torch.numel(xt[0, :])
            J = torch.reshape(J, (J.shape[0], tot_size, tot_size)) # of size batch x x_dim x x_dim
            id_batch = torch.eye(tot_size, device=xt.device).expand(J.shape)
            cov_matrix_unscaled = id_batch + (sigma*sigma)*(J)
            #cov_matrix_unscaled = self._check_spd(cov_matrix_unscaled)
            cov_matrix = (sigma*sigma) * (cov_matrix_unscaled)
            #logging.info("The relative norm of the Jacobian is {:.4f}".format(sigma* torch.linalg.norm(J)/math.sqrt(tot_size*J.shape[0])))
            cov_inv = torch.linalg.inv(cov_matrix)
            elapsed_time = time.time() - init_time
            #logging.info("Additional computations took {:.4f} seconds".format(elapsed_time))
            #return lambda x: (torch.matmul(cov_inv, x.reshape(xt.shape[0], tot_size, 1)).reshape(xt.shape))
            return cov_inv
        else:
            raise Exception("Covariance inverse operator type not coded yet!")

    def _check_spd(self, cov_matrix_unscaled, epsilon=10**(-3)):
        try:
            L, V = torch.linalg.eig(cov_matrix_unscaled)
            test = torch.real(V) @ torch.diag_embed(torch.clamp(torch.real(L), min=epsilon)) @ torch.linalg.inv(torch.real(V))
            return test 
        except:
            logging.info("Oh no! Correction failed")
            return cov_matrix_unscaled

    def _check(self, annealing_scheduler_config, diffusion_scheduler_config):
        # sigma_max of diffusion scheduler change each step
        if 'sigma_max' in diffusion_scheduler_config:
            diffusion_scheduler_config.pop('sigma_max')

        # sigma final of annealing scheduler should always be 0
        annealing_scheduler_config['sigma_final'] = 0
        return annealing_scheduler_config, diffusion_scheduler_config

    def _record(self, xt, x0y, x0hat, sigma, beta, x0hat_results, x0y_results):
        """
            Records the intermediate states during optimization.
        """
        self.trajectory.add_tensor(f'xt', xt)
        self.trajectory.add_tensor(f'x0y', x0y)
        self.trajectory.add_tensor(f'x0hat', x0hat)
        self.trajectory.add_value(f'sigma', sigma)
        self.trajectory.add_value(f'beta', beta)
        if x0hat_results is not None:
            for name in x0hat_results.keys():
                self.trajectory.add_value(f'x0hat_{name}', x0hat_results[name])
        if x0y_results is not None:
            for name in x0y_results.keys():
                self.trajectory.add_value(f'x0y_{name}', x0y_results[name])
    
    def get_start(self, ref):
        x_start = torch.randn_like(ref) * self.annealing_scheduler.sigma_max
        return x_start

