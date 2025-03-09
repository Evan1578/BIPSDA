import warnings
import torch
from abc import ABC, abstractmethod
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt

__OPERATOR__ = {}


def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            if __OPERATOR__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __OPERATOR__[name] = cls
        cls.name = name
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class Operator(ABC):
    """
    Abstract operator class
    Based on code in this repository: https://github.com/zhangbingliang2019/DAPS/tree/fb67270816876a3229b7768c18c8ab9cd2c0c10f
    """
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def measure(self, x):
        pass

    @abstractmethod
    def log_likelihood(self, x, y):
        pass

    def likelihood(self, x, y):
        return torch.exp(self.log_likelihood(x, y))


# operators

@register_operator(name='inpainting1D')
class Inpainting1D(Operator):
    def __init__(self, data_dim, observed_indices, sigma=0.05):
        """
        Inputs: 
        A (torch.tensor): measurement matrix (of size observation_dim x unknown_dim)
        mask (torch.tensor): binary-valued mask corresponding to A (of size unknown_dim)
        sigma (positive scalar): standard deviation of noise
        """
        self.meas_dim = meas_dim = len(observed_indices)
        self.data_dim = data_dim
        self.A = A = torch.zeros(meas_dim, data_dim)
        for idx, num in enumerate(observed_indices):
            A[idx, num] = 1
        self.At = A.transpose(0, 1) # size unknown_dim x observation_dim
        all_indices = [i for i in range(A.shape[1])]
        self.unobserved_indices = torch.tensor([x for x in all_indices if x not in observed_indices], device=A.device, dtype=torch.long)
        self.observed_indices = torch.tensor(observed_indices, device=A.device, dtype=torch.long)
        self.device = A.device
        self.sigma = sigma
        super().__init__()
    
    def __call__(self, x):
        x_device = x.device
        return (x.to(self.device) @ self.At).to(x_device)
    
    def log_likelihood(self, x, y):
        error = ((self(x) - y) ** 2).flatten(1).sum(-1)
        return -error / 2 /self.sigma ** 2
    
    def measure(self, x):
        y0 = self(x)
        return y0 + self.sigma * torch.randn_like(y0)
    
    def solve_proximal(self, x0hat, measurement, cinv, cov_type, record=False, verbose=False, params=None):
        if cov_type == 'identity':
            nullspace_comp = x0hat[:, self.unobserved_indices] # (1 - self.mask) is ones where measurements are not taken, zero where measurements are
            data_informed_comp = (cinv * x0hat[:, self.observed_indices] + measurement/(self.sigma * self.sigma))/(cinv + 1/(self.sigma*self.sigma))
            full_sol = torch.zeros_like(x0hat)
            full_sol[:, self.unobserved_indices] = nullspace_comp
            full_sol[:, self.observed_indices] = data_informed_comp
            return full_sol
        elif cov_type == 'exact':
            A = self.A.to(x0hat.device)
            At =self.At.to(x0hat.device)
            lhs = cinv + ((At@A)/(self.sigma*self.sigma))[None, :, :]
            rhs =  (measurement@A)/(self.sigma * self.sigma) + torch.squeeze(torch.matmul(cinv, torch.unsqueeze(x0hat, dim=-1)))
            full_sol = torch.linalg.solve(lhs, rhs)
            return full_sol
        else:
            raise Exception("Unknown Covariance Type!")

@register_operator(name='gaussphaseretrieval1D')
class GaussianPhaseRerieval1D(Operator):

    def __init__(self, dims, seed=None, sigma=0.05):
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        self.A = torch.randn(dims[0], dims[1], generator=generator)
        self.At = self.A.transpose(0, 1)
        self.meas_dim = dims[0]
        self.data_dim = dims[1]
        self.sigma = sigma
        super().__init__()

    def __call__(self, x):
        if x.device != self.At.device:
            self.At = self.At.to(x.device)
        projected_data = x @ self.At
        return projected_data**2
    
    def log_likelihood(self, x, y):
        error = ((self(x) - y) ** 2).flatten(1).sum(-1)
        return -error / 2 /self.sigma ** 2
    
    def measure(self, x):
        y0 = self(x)
        return y0 + self.sigma * torch.randn_like(y0)
    
    def form_gauss_newton_hessian(self, precision, x):
        Ax = torch.squeeze(torch.matmul(torch.unsqueeze(self.A.to(x.device), dim=0), torch.unsqueeze(x, dim=-1)))
        Ax_diag = torch.diag_embed(Ax)
        Jacob_fwd_model = 2 * torch.matmul(torch.unsqueeze(self.At.to(x.device), dim=0), Ax_diag)
        batch_gauss_newton_hessian = (1 / (self.sigma * self.sigma)) * torch.matmul(Jacob_fwd_model, torch.transpose(Jacob_fwd_model, 1, 2))
        return batch_gauss_newton_hessian + precision
        
    

@register_operator(name='xray_tomography')
class XrayOperator(Operator):

    def __init__(self, A, I0):
        self.I0 = I0
        self.A = A
        self.At = A.transpose(0, 1)
        self.meas_dim =A.shape[0]
        self.data_dim = A.shape[1]
        super().__init__()

    def __call__(self, x):
        if x.device != self.At.device:
            self.At = self.At.to(x.device)
        projected_data = (x +10.) @ self.At
        log_of_term = math.log(self.I0) -1.*projected_data
        return torch.exp(log_of_term)

    def measure(self, x):
        y0 = self(x)
        return torch.poisson(y0, generator=None)

    def log_likelihood(self, x, y):
        x_pred = self(x)
        # log_y_facterm = .5*torch.log(2*math.pi*y) + y*(torch.log(y) - 1.) # using sterling's approximation
        comp_like = y * torch.log(x_pred) - x_pred #- log_y_facterm
        return torch.sum(comp_like, dim=-1)
    
