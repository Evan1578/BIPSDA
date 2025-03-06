"""
Run evaluation for a given trial 
"""

import os
import pickle
import random
import math

import numpy as np
import logging
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


import distributions
import forward_operators
from eval_metrics import RBF, cmd, batch_mmd_loss


def comp_generative_w_gt(generative_samples, ground_truth_samples, fout, labels=['Generative Samples', 'Ground Truth Samples'], proj_matrix=None, 
                         title="2D Projections of Samples", colors=['#1f77b4', '#ff7f0e']):
    """
    generative_samples (torch.Tensor): tensor of size batch_size x D containing generative modeling samples
    ground_truth_samples (torch.Tensor): tensor of size batch_size x D containing ground truth samples
    fout (str): file to save plot to 
    """
    assert generative_samples.shape == ground_truth_samples.shape # dimensions should be the same
    # generate random projection matrix
    if proj_matrix is None:
        proj_matrix = torch.randn(generative_samples.shape[1], 2)
    # take projection
    proj_gen_samples  = torch.matmul(generative_samples, proj_matrix)
    proj_gt_samples = torch.matmul(ground_truth_samples, proj_matrix)
    # plot results
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_title(title, fontsize=20)
    ax.scatter(torch.minimum(torch.maximum(proj_gen_samples[:, 0], torch.tensor([-100])), torch.tensor([100])), 
               torch.minimum(torch.maximum(proj_gen_samples[:, 1], torch.tensor([-100])), torch.tensor([100])), label=labels[0], c=colors[0])
    ax.scatter(torch.minimum(torch.maximum(proj_gt_samples[:, 0], torch.tensor([-100])), torch.tensor([100])), 
               torch.minimum(torch.maximum(proj_gt_samples[:, 1], torch.tensor([-100])), torch.tensor([100])), label=labels[1], c=colors[1])
    ax.legend(fontsize=14)
    ax.grid(True)
    plt.savefig(fout)
    plt.close()

def plot_trajectory(trajectory, proj_matrix, save_dir):
        for idx in range(0, len(trajectory), 5):
            fout = os.path.join(save_dir, 'idx_' + str(idx) + '.png')
            data = trajectory[idx]
            fig, ax = plt.subplots(figsize=(9, 9))
            projected_data = torch.matmul(data, proj_matrix)
            ax.scatter(torch.minimum(torch.maximum(projected_data[:, 0], torch.tensor([-50])), torch.tensor([50])), torch.minimum(torch.maximum(projected_data[:, 1], torch.tensor([-50])), torch.tensor([50])))
            ax.grid(True)
            plt.savefig(fout)
            plt.close()

def histogram_comp(samples, log_prob_fn, fout, range = None, color=None):
    log_probs = log_prob_fn(samples)
    #print("The minimum log probability was {:.4f}".format(torch.min(log_probs).item()))
    #print("The mmaximum log probability was {:.4f}".format(torch.max(log_probs).item()))
    fig, axs = plt.subplots()
    axs.hist(log_probs, bins=100, range=range, color=color, density=True)
    axs.set_xlabel('Log Probability', fontsize=18)
    axs.set_ylabel('Density', fontsize=18)
    axs.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axs.tick_params(labelsize=12)
    plt.savefig(fout, dpi=300, bbox_inches = "tight")
    plt.close()


def run_eval(eval_ops, sampledir, gtdir, eval_name):


    # set seeds
    torch.manual_seed(eval_ops.seed)
    np.random.seed(eval_ops.seed)
    random.seed(eval_ops.seed)
 
    # create directory for evaluation results
    evaldir = os.path.join(sampledir, eval_name)
    os.mkdir(evaldir)

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
    with open(os.path.join(sampledir, 'sample_results.pickle'), "rb") as handle:
        results = pickle.load(handle)
    with open(os.path.join(sampledir, 'args.pickle'), "rb") as handle:
        args = pickle.load(handle)
    with open(os.path.join(sampledir, 'config.pickle'), "rb") as handle:
        configs = pickle.load(handle)
    with open(os.path.join(gtdir, "sample_results.pickle"), "rb") as handle:
        gt_results = pickle.load(handle)

    # some sanity checks
    plot_options = eval_ops.plot_options.lower()
    assert plot_options in {'first', 'none', 'all', 'first10'}

    # form prior and forward operator
    args.model_dir = 'checkpoints/GaussMixture/Energy'
    with open(os.path.join(args.model_dir, 'config.pickle'), 'rb') as handle:
        model_config = pickle.load(handle)
    prior = distributions.create_distribution(model_config)

    operator = forward_operators.get_operator(**configs.operator) # configs should all have the same operator
    meas_dim = operator.meas_dim

    # form projection matrices for visualization (assume observable subspace determined by linear operator on unknown variable)
    _, _, Vt = torch.linalg.svd(operator.A)
    Vt_observed = Vt[0:meas_dim, :]
    Vt_unobserved = Vt[meas_dim:, :]

    if configs.operator.name == 'xray_tomography':
        proj_matrix_one = torch.zeros((10, 2))
        proj_matrix_one[0, 0] = proj_matrix_one[1, 1] = 1
        proj_matrix_two = torch.zeros((10, 2))
        proj_matrix_two[2, 0] = proj_matrix_two[3, 1] = 1

    # loop through results and plot results, compute metrics
    means_tot = []
    vars_tot = []
    discrepancies_tot = []
    total_discarded = 0
    num_with_discarded = 0
    if plot_options in {'first', 'all', 'first10'}:
        plotdir = os.path.join(evaldir, 'Plots')
        os.mkdir(plotdir)
    for idx, result in enumerate(results):
        # get results from dictionary
        measurement = result["measurement"] 
        assert torch.allclose(measurement, gt_results[idx]["measurement"]) # sanity check
        gt_samples = gt_results[idx]['gt_samples']
        gt_samples2 = gt_results[idx]['gt_samples2']
        # form prior, likelihood, and posterior log probability functions
        prior_log_prob = lambda x: prior.log_prob(x)
        likelihood_log_prob = lambda x: operator.log_likelihood(x, measurement) #- (meas_dim/2)*math.log(math.pi*2) - meas_dim*math.log(operator.sigma)
        posterior_log_prob = lambda x : likelihood_log_prob(x) + prior_log_prob(x)
        # plot results for given trial
        if (idx == 0 and plot_options == 'first') or plot_options == 'all' or (idx < 10 and plot_options == 'first10'):
            logging.info("Plotting results from " + str(idx) + "th trial ...")
            samples = result['samples']
            comp_generative_w_gt(samples[:eval_ops.num_plot, :], gt_samples[:eval_ops.num_plot, :], os.path.join(plotdir, 'daps_vs_gt_comps1_' + str(idx) + '.png'), 
                                    proj_matrix=proj_matrix_one, colors=['#ff7f0e', '#1f77b4'], title='Two Random Components')
            comp_generative_w_gt(samples[:eval_ops.num_plot, :], gt_samples[:eval_ops.num_plot, :], os.path.join(plotdir, 'daps_vs_gt_comps2_' + str(idx) + '.png'), 
                                    proj_matrix=proj_matrix_two, colors=['#ff7f0e', '#1f77b4'], title='Two More Random Componecnnts')
            comp_generative_w_gt(samples[:eval_ops.num_plot, :], gt_samples[:eval_ops.num_plot, :], os.path.join(plotdir, 'daps_vs_gt_random_' + str(idx) + '.png'))
            histogram_comp(samples, prior_log_prob, os.path.join(plotdir, 'hist_daps_prior_' + str(idx) + '.png'), color='#ff7f0e')
            histogram_comp(samples, likelihood_log_prob, os.path.join(plotdir, 'hist_daps_like_' + str(idx) + '.png'), color='#ff7f0e')
            histogram_comp(samples, posterior_log_prob, os.path.join(plotdir, 'hist_daps_post_' + str(idx) + '.png'), color='#ff7f0e')
            histogram_comp(gt_samples, prior_log_prob, os.path.join(plotdir, 'hist_gt_prior_' + str(idx) + '.png'), color='#1f77b4')
            histogram_comp(gt_samples, likelihood_log_prob, os.path.join(plotdir, 'hist_gt_like_' + str(idx) + '.png'), color='#1f77b4')
            histogram_comp(gt_samples, posterior_log_prob, os.path.join(plotdir, 'hist_gt_post_' + str(idx) + '.png'), color='#1f77b4')

        # compute metrics
        logging.info("Conductive quantitative tests on results from " + str(idx) + "th trial ...")
        means = {}
        vars = {}
        discrepancies = {}
        # form mmd and cmd metrics
        size_subset = 1000
        if hasattr(eval_ops, 'mmd_kernel_bandwidth'):
            kernel_bandwidth = eval_ops.mmd_kernel_bandwidth
        else:
            cross_dist = torch.cdist(gt_samples[:size_subset, :], gt_samples[:size_subset, :])
            kernel_bandwidth = (cross_dist.data.sum() / (size_subset ** 2 - size_subset)).item()
        corr_sigma = math.sqrt(.5/kernel_bandwidth)
        mmd_kernel = RBF(bandwidth = kernel_bandwidth, mul_factor=2., n_kernels=5)
        if hasattr(eval_ops, 'cmd_decay_rate'):
            b = eval_ops.cmd_decay_rate
        else:
            b = 4. * torch.max(torch.std(gt_samples, dim=0)).item()
        if idx == 0:
            logging.info("Value of (b-a) used in CMD kernel calculation was {:.4f}".format(b))
            logging.info("RBF kernel implemented with {:d} kernels, {:.4f} multiplication factor, and bandwidth corresponding to sigma = {:.4f}".format(5, 2, corr_sigma))
        # discard samples that escape from score model domain
        samples = result['samples']
        maxsample, _ = torch.max(samples, dim=-1) 
        minsample, _ = torch.min(samples, dim=-1)
        mask = torch.logical_and((maxsample < 100 ), (minsample > - 100.))
        num_discard = (samples.shape[0] - torch.count_nonzero(mask)).item()
        logging.info("Number discarded was {:d}".format(num_discard))
        if num_discard > 0:
            num_with_discarded += 1
        total_discarded += num_discard
        samples = samples[mask]
        # comput cmd and mmd errors
        discrepancies['cmd'] = cmd_error = cmd(samples, gt_samples, b=b, a=0.)
        discrepancies['mmd'] = mmd_loss = batch_mmd_loss(samples, gt_samples, mmd_kernel)
        # compute mean errors
        posterior_mean = torch.mean(gt_samples2, dim=0)
        means['total'] = mean_error = torch.linalg.norm(posterior_mean - torch.mean(samples, dim=0)).item()
        means['unobserved'] = mean_error_unobserved = torch.linalg.norm(Vt_unobserved @ (posterior_mean - torch.mean(samples, dim=0))).item()
        means['observed'] = mean_error_observed = torch.linalg.norm(Vt_observed @ (posterior_mean - torch.mean(samples, dim=0))).item()
        # compute variance errors
        posterior_pointwise_var = torch.var(gt_samples2, dim=0)
        var = torch.var(samples, dim=0)
        var_unobserved = Vt_unobserved @ var
        var_observed = Vt_observed @ var
        vars['total']= var_error = torch.linalg.norm(posterior_pointwise_var - var).item()
        vars['unobserved'] = var_error_unobserved = torch.linalg.norm(Vt_unobserved @ posterior_pointwise_var - var_unobserved).item()
        vars['observed'] = var_error_observed = torch.linalg.norm(Vt_observed @ posterior_pointwise_var - var_observed).item()
        # print results
        logging.info("The mean errors for this trial were: {:.4f} (total), {:.4f} (observed components), {:.4f} (unobserved, components)"
                .format( mean_error, mean_error_observed, mean_error_unobserved))
        logging.info("The variance errors for GIPSDA:0 were: {:.4f} (total), {:.4f} (observed components), {:.4f} (unobserved components)"
                .format( var_error, var_error_observed, var_error_unobserved))
        logging.info("The mean variances were {:.4f} (total), {:.4f} (observed components), {:.4f} (unobserved components)"
                .format( torch.mean(var).item(), torch.mean(var_observed).item(), torch.mean(var_unobserved).item()))
        logging.info("The cmd error (K = 5) was {:.4f}".format(cmd_error))
        logging.info("The mmd error was {:.4f}".format(mmd_loss))
        # report discrepancies between two i.i.d. sets of gt samples
        discrepancies['Exact_cmd'] = cmd_error = cmd(gt_samples, gt_samples2, b=b, a=0.)
        discrepancies['Exact_mmd'] = mmd_loss = batch_mmd_loss(gt_samples, gt_samples2, mmd_kernel)
        means['Exact_total'] = mean_error = torch.linalg.norm(posterior_mean - torch.mean(gt_samples, dim=0)).item()
        means['Exact_unobserved']= mean_error_unobserved = torch.linalg.norm(Vt_unobserved @ (posterior_mean - torch.mean(gt_samples, dim=0))).item()
        means['Exact_observed'] = mean_error_observed = torch.linalg.norm(Vt_observed @ (posterior_mean - torch.mean(gt_samples, dim=0))).item()
        logging.info("The mean errors for the exact samples were: {:.4f} (total), {:.4f} (observed components), {:.4f} (unobserved components)"
                .format( mean_error, mean_error_observed, mean_error_unobserved))
        var = torch.var(gt_samples, dim=0)
        var_unobserved = Vt_unobserved @ var
        var_observed = Vt_observed @ var
        vars['Exact'] = var_error = torch.linalg.norm(posterior_pointwise_var - var).item()
        vars['Exact_unobserved'] = var_error_unobserved = torch.linalg.norm(Vt_unobserved @ posterior_pointwise_var - var_unobserved).item()
        vars['Exact_observed'] = var_error_observed = torch.linalg.norm(Vt_observed @ posterior_pointwise_var - var_observed).item()
        logging.info("The variance errors for the exact samples were: {:.4f} (total), {:.4f} (observed components), {:.4f} (unobserved components)"
                .format(var_error, var_error_observed, var_error_unobserved))
        logging.info("The mean variances for the exact samples were {:.4f} (total), {:.4f} (observed components), {:.4f} (unobserved components)"
                .format( torch.mean(var).item(), torch.mean(var_observed).item(), torch.mean(var_unobserved).item()))
        logging.info("The cmd error for the exact samples (K = 5) was {:.4f}".format(cmd_error))
        logging.info("The mmd loss for the exact samples was {:.4f}".format(mmd_loss))
        # Append trial results to final results
        means_tot.append(means)
        vars_tot.append(vars)
        discrepancies_tot.append(discrepancies)
    logging.info("Discarded {:d} total samples out of 1 million; {:d} trials out of 100 had at least 1 sample discarded".format(total_discarded, num_with_discarded))
    discarded = {'total_discarded': total_discarded, 'num_with_discarded':num_with_discarded}
    with open(os.path.join(evaldir, 'means.pickle'), "wb") as handle:
        pickle.dump(means_tot, handle)
    with open(os.path.join(evaldir, 'variances.pickle'), "wb") as handle:
        pickle.dump(vars_tot, handle)
    with open(os.path.join(evaldir, 'discrepancies.pickle'), "wb") as handle:
        pickle.dump(discrepancies_tot, handle)
    with open(os.path.join(evaldir, 'discarded.pickle'), "wb") as handle:
        pickle.dump(discarded, handle)
    logging.info('Evaluation finished without errors')
