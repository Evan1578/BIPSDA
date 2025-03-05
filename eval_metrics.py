import torch
from torch import nn
import logging
import pickle
import os
import numpy as np
import random
import tensorflow as tf
import math
import matplotlib.pyplot as plt

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth
    
    def forward_diff(self, X, Y):
        L2_distances = torch.cdist(X, Y) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)
    
class MMDLoss(nn.Module):
    """"
    Implementatiuon from: https://github.com/yiftachbeer/mmd_loss_pytorch/blob/756d20f4c7f133b11ca1f2cc0834b411a5e91db9/mmd_loss.py
    """

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY
    

def cmd(x1, x2, n_moments=5, b=None, a=None):
    mx1 = torch.mean(x1, dim=0)
    mx2 = torch.mean(x2, dim=0)
    if b is None:
        b = max(torch.max(x1), torch.max(x2))
    if a is None:
        a = min(torch.min(x1), torch.min(x2))
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = torch.sqrt(torch.sum(((mx1-mx2)**2))) / (b - a)
    scm = dm
    for i in range(n_moments - 1):
        ss1 = torch.mean((sx1**(i+2)), dim=0)
        ss2 = torch.mean((sx2**(i+2)), dim=0)
        scm += torch.sqrt(torch.sum(((ss1-ss2)**2))) / (b - a)**(i+2)
    return scm

def batch_kernel(array1, array2, kernel, batch_size):

    if array1.shape[0]*array2.shape[0] < batch_size: # compute directly if possible
        return kernel.forward_diff(array1, array2).mean()
    else: # otherwise divide both arrays into two and compute recursively
        div_array1 = array1.shape[0] // 2
        div_array2 = array2.shape[0] // 2
        part1 = batch_kernel(array1[: div_array1, :], array2[:div_array2, :], kernel, batch_size)*(div_array1*div_array2)
        part2 = batch_kernel(array1[: div_array1, :], array2[div_array2:, :], kernel, batch_size)*(div_array1*(array2.shape[0] - div_array2))
        part3 = batch_kernel(array1[ div_array1:, :], array2[:div_array2, :], kernel, batch_size)*((array1.shape[0] - div_array1)*div_array2)
        part4 = batch_kernel(array1[ div_array1:, :], array2[div_array2:, :], kernel, batch_size)*((array1.shape[0] - div_array1)*(array2.shape[0] - div_array2))
        return (part1 + part2 + part3 + part4) / (array1.shape[0]*array2.shape[0])


def batch_mmd_loss(x1, x2, kernel, batch_size=1e6):

    x1_corr = batch_kernel(x1, x1, kernel, batch_size)
    x2_corr = batch_kernel(x2, x2, kernel, batch_size)
    cross_corr = batch_kernel(x1, x2, kernel, batch_size)
    estimate = x1_corr + x2_corr - 2. * cross_corr

    return estimate


def linear_mmd_loss(x1, x2, kernel):
    # NOTE: not tested yet, kernel has to have straightforward fwd meaning
    assert x1.shape[0] == x2.shape[0]
    assert x1.shape[0] % 2 == 0
    m2 = x1.shape[0] // 2
    estimate = 0
    z = torch.stack(x1, x2)
    z_even = z[:, 0:2:, :]
    z_odd = z[:, 1:2:, :]
    for i in range(m2):
        val1 = z_even[:, i, :]
        val2 = z_odd[:, i, :]
        estimate += kernel(val1[0, :], val2[0, :]) + kernel(val1[1, :], val2[1, :]) -  kernel(val1[0, :], val2[1, :]) - kernel(val1[1, :], val2[0, :])
    return estimate / m2

def test_mmd_metric(eval_ops, resultsfolder, eval_name):
    # set seeds
    torch.manual_seed(eval_ops.seed)
    np.random.seed(eval_ops.seed)
    random.seed(eval_ops.seed)

    # create directory for evaluation results
    workdir = os.path.join('results', resultsfolder)
    evaldir = os.path.join(workdir, eval_name)
    tf.io.gfile.makedirs(evaldir)


    # initialize logger
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(evaldir, ".log")), #TODO: change
        logging.StreamHandler()
    ]
    )

    # load results and experiment configurations
    with open(os.path.join(workdir, 'sample_results.pickle'), "rb") as handle:
        results = pickle.load(handle)
    
    # first verify that batch mmd loss works 
    logging.info('Testing batch implementation of mmd loss ...')
    mmd_kernel = RBF(bandwidth = 1., mul_factor=1., n_kernels=1)
    x1 = results[0]['gt_samples'][:1000, :]
    x2 = results[0]['bipsda:0'][:1000, :]
    mmd_loss_official = MMDLoss(mmd_kernel)
    official_mmd = mmd_loss_official(x1, x2)
    my_mmd = batch_mmd_loss(x1, x2, mmd_kernel, batch_size=1e7)
    logging.info("The official mmd loss is {:.4f}".format(official_mmd))
    logging.info("My batch mmd loss gave an mmd loss of {:.4f}".format(my_mmd))

    # load results 
    measurement = results[0]["measurement"] 
    gt_samples = results[0]['gt_samples']
    gt_samples2 = results[0]['gt_samples2']
    gispda0_samples = results[0]['bipsda:0']
    gispda1_samples = results[0]['bipsda:1']

    # test as function of number of samples
    size_subset = 1000
    cross_dist = torch.cdist(gt_samples[:size_subset, :], gt_samples[:size_subset, :])
    kernel_bandwidth = cross_dist.data.sum() / (size_subset ** 2 - size_subset)
    corr_sigma = math.sqrt(.5/kernel_bandwidth.item())
    num_samples = torch.logspace(1, math.log10(gt_samples.shape[0]), 5)
    logging.info('Testing as function of number of samples (n_kernels=1, correlation length = {:.4f})'.format(corr_sigma))
    stats = np.zeros((3, len(num_samples)))
    for idx, n_sample in enumerate(num_samples):
        n_sample = int(n_sample)
        logging.info('The number of samples is {:d}'.format(n_sample))
        mmd_kernel = RBF(bandwidth = kernel_bandwidth, mul_factor=2., n_kernels=1)
        mmd_loss_official = MMDLoss(mmd_kernel)
        stats[0, idx] = bipsda0_mmd = mmd_loss_official(gispda0_samples[:n_sample, :], gt_samples[:n_sample, :])
        stats[1, idx] = bipsda1_mmd = mmd_loss_official(gispda1_samples[:n_sample, :], gt_samples[:n_sample, :])
        stats[2, idx] = gt_mmd = mmd_loss_official(gt_samples2[:n_sample, :], gt_samples[:n_sample, :])
        logging.info("The mmd error for BIPSDA:0 (DAPS) was {:.4f}".format(bipsda0_mmd))
        logging.info("The mmd error for BIPSDA:1 (DiffPIR) was {:.4f}".format(bipsda1_mmd))
        logging.info("The mmd error between two sets of i.i.d. samples from ground truth posterior was {:.4f}".format(gt_mmd))
    plt.plot(num_samples, stats[0, :], label='BIPSDA:0 (DAPS)')
    plt.plot(num_samples, stats[1, :], label='BIPSDA:1 (DiffPIR)')
    plt.plot(num_samples, stats[2, :], label='GT')
    plt.title('MMD metric')
    plt.xlabel('Number of Samples')
    plt.ylabel('Discrepancy')
    plt.legend()
    plt.savefig(os.path.join(evaldir, 'n_samples.png'))
    plt.close()


    # test as function of correlation length
    corr_sigmas = torch.linspace(.1, 2, 10)
    logging.info('Testing as function of correlation length in kernel (n_kernels=1)')
    stats = np.zeros((3, len(corr_sigmas)))
    for idx, corr_sigma in enumerate(corr_sigmas):
        logging.info('The correlation length is {:.4f}'.format(corr_sigma))
        mmd_kernel = RBF(bandwidth = 1/(2*corr_sigma*corr_sigmas), mul_factor=2., n_kernels=1)
        mmd_loss_official = MMDLoss(mmd_kernel)
        stats[0, idx] = bipsda0_mmd = mmd_loss_official(gispda0_samples, gt_samples)
        stats[1, idx] = bipsda1_mmd = mmd_loss_official(gispda1_samples, gt_samples)
        stats[2, idx] = gt_mmd = mmd_loss_official(gt_samples2, gt_samples)
        logging.info("The mmd error for BIPSDA:0 (DAPS) was {:.4f}".format(bipsda0_mmd))
        logging.info("The mmd error for BIPSDA:1 (DiffPIR) was {:.4f}".format(bipsda1_mmd))
        logging.info("The mmd error between two sets of i.i.d. samples from ground truth posterior was {:.4f}".format(gt_mmd))
    plt.plot(corr_sigmas, stats[0, :], label='BIPSDA:0 (DAPS)')
    plt.plot(corr_sigmas, stats[1, :], label='BIPSDA:1 (DiffPIR)')
    plt.plot(corr_sigmas, stats[2, :], label='GT')
    plt.title('MMD metric')
    plt.xlabel('Correlation Length')
    plt.ylabel('Discrepancy')
    plt.legend()
    plt.savefig(os.path.join(evaldir, 'corr_length.png'))
    plt.close()


    # test as function of n kernels 
    size_subset = 1000
    cross_dist = torch.cdist(gt_samples[:size_subset, :], gt_samples[:size_subset, :])
    kernel_bandwidth = cross_dist.data.sum() / (size_subset ** 2 - size_subset)
    corr_sigma = math.sqrt(.5/kernel_bandwidth.item())
    nkernels = [1, 2, 3, 4, 5, 6, 7,]
    logging.info('Testing as function of n kernels with sigma in kernel = {:.4f}'.format(corr_sigma))
    stats = np.zeros((3, len(nkernels)))
    for idx, nkernel in enumerate(nkernels):
        logging.info('The number of kernels is {:d}'.format(nkernel))
        mmd_kernel = RBF(bandwidth = kernel_bandwidth, mul_factor=2., n_kernels=nkernel)
        mmd_loss_official = MMDLoss(mmd_kernel)
        stats[0, idx] = bipsda0_mmd = mmd_loss_official(gispda0_samples, gt_samples)
        stats[1, idx] = bipsda1_mmd = mmd_loss_official(gispda1_samples, gt_samples)
        stats[2, idx] = gt_mmd = mmd_loss_official(gt_samples2, gt_samples)
        logging.info("The mmd error for BIPSDA:0 (DAPS) was {:.4f}".format(bipsda0_mmd))
        logging.info("The mmd error for BIPSDA:1 (DiffPIR) was {:.4f}".format(bipsda1_mmd))
        logging.info("The mmd error between two sets of i.i.d. samples from ground truth posterior was {:.4f}".format(gt_mmd))
    plt.plot(nkernels, stats[0, :], label='BIPSDA:0 (DAPS)')
    plt.plot(nkernels, stats[1, :], label='BIPSDA:1 (DiffPIR)')
    plt.plot(nkernels, stats[2, :], label='GT')
    plt.title('MMD metric')
    plt.xlabel('Number of Kernels')
    plt.ylabel('Discrepancy')
    plt.legend()
    plt.savefig(os.path.join(evaldir, 'nkernels.png'))
    plt.close()


def test_cmd_metric(eval_ops, resultsfolder, eval_name):

    # set seeds
    torch.manual_seed(eval_ops.seed)
    np.random.seed(eval_ops.seed)
    random.seed(eval_ops.seed)

    # create directory for evaluation results
    workdir = os.path.join('results', resultsfolder)
    evaldir = os.path.join(workdir, eval_name)
    tf.io.gfile.makedirs(evaldir)


    # initialize logger
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(evaldir, ".log")), #TODO: change
        logging.StreamHandler()
    ]
    )

    # load results and experiment configurations
    with open(os.path.join(workdir, 'sample_results.pickle'), "rb") as handle:
        results = pickle.load(handle)

    for idx, result in enumerate(results):
        # get results from dictionary
        measurement = result["measurement"] 
        gt_samples = result['gt_samples']
        gt_samples2 = result['gt_samples2']
        gispda0_samples = result['bipsda:0']
        gispda1_samples = result['bipsda:1']
        stds = [2, 3, 4, 5, 6, 7]
        Ks = [1, 2, 3, 4, 5, 6, 7, 8]
        range_gt = torch.max(gt_samples).item() - torch.min(gt_samples).item()
        logging.info("The range of the GT data is {:.4f}".format(range_gt))
        for std in stds:
            stats = np.zeros((3, len(Ks)))
            b = std * torch.max(torch.std(gt_samples, dim=0)).item()
            a = 0.
            logging.info("Value of (b-a) used was {:d} * std of gt data; this corresponds to {:.4f}".format(std, b))
            for idx, K in enumerate(Ks):
                logging.info("Number of moments used was {:d}".format(K))
                stats[0, idx] = bipsda0_cmd = cmd(gispda0_samples, gt_samples, b=b, a=a, n_moments=K)
                stats[1, idx] = bipsda1_cmd = cmd(gispda1_samples, gt_samples, b=b, a=a, n_moments=K)
                stats[2, idx] = gt_cmd = cmd(gt_samples2, gt_samples, b=b, a=a, n_moments=K)
                logging.info("The cmd error for BIPSDA:0 (DAPS) was {:.4f}".format(bipsda0_cmd))
                logging.info("The cmd error for BIPSDA:1 (DiffPIR) was {:.4f}".format(bipsda1_cmd))
                logging.info("The cmd error between two sets of i.i.d. sampeles from ground truth posterior was {:.4f}".format(gt_cmd))
            plt.plot(Ks, stats[0, :], label='BIPSDA:0 (DAPS)')
            plt.plot(Ks, stats[1, :], label='BIPSDA:1 (DiffPIR)')
            plt.plot(Ks, stats[2, :], label='GT')
            plt.title('CMD metric (num stds = {:d})'.format(std))
            plt.legend()
            plt.savefig(os.path.join(evaldir, 'std_' + str(std) + '.png'))
            plt.close()

