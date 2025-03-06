import os
import math

import torch
import logging
import pickle
import numpy as np
import random

import distributions
from distributions import get_mixture_gaussian_posterior, get_convolved_distribution
import score_model
from score_model import restore_checkpoint, ExponentialMovingAverage
from utils import Parameter, Likelihood, extract_samples_from_chain, mix_to_device
import score_model
import pydream 
from pydream import core
from pydream import convergence
import forward_operators
from diffusion_solver import get_solver


def pydream_sample(workdir, prior, operator, measurement, nchains=5, its=10000, nsamples=1000, seed_history=True, is_gaussian=True):
    # form Pydream-compatible likelihood and prior
    parameter = Parameter(prior, operator.data_dim)
    likelihood = Likelihood(operator, measurement, is_gaussian=is_gaussian)
    # seed history
    def Latin_hypercube(minn, maxn, N):
        y = np.random.rand(N, len(minn))
        x = np.zeros((N, len(minn)))
        
        for j in range(len(minn)):
            idx = np.random.permutation(N)
            P = (idx - y[:,j])/N
            x[:,j] = minn[j] + P * (maxn[j] - minn[j])
        return x
    if seed_history:
        history = Latin_hypercube(np.linspace(-10, -10, num=10), np.linspace(10, 10, num=10), 10000)
        np.save(os.path.join(workdir, 'history_file.npy'), history)
        start = [history[chain] for chain in range(nchains)]
        # run pyDream
        params, log_ps = core.run_dream([parameter], likelihood, nchains=nchains, niterations=its, save_history=False, 
                                        verbose=False, start=start, start_random=False, history_file = os.path.join(workdir, 'history_file.npy'))
    else:
        params, log_ps = core.run_dream([parameter], likelihood, nchains=nchains, niterations=its, save_history=False, verbose=False, start_random=True)
    # report summary statistics
    GR = convergence.Gelman_Rubin(params)
    logging.info("The maximum GR value was {:.4f}".format(np.max(GR)))
    logging.info("The minimum and maximum log_p (from one chain, discarding burn in) is {:.6f}, {:.6f}".format(np.min(log_ps[0][its//2:]), np.max(log_ps[0][its//2:])))
    # extract samples from chains
    samples = extract_samples_from_chain(params, nsamples)
    return samples

def get_log_likes(gaussian, operator, measurement, nsamples):
    log_likes = np.zeros(nsamples)
    prior_samples = gaussian.sample(num_samples=nsamples)
    measurement_repeated = measurement.repeat(nsamples, 1)
    log_likes_unnorm = operator.log_likelihood(prior_samples, measurement_repeated)
    log_likes = log_likes_unnorm - (operator.meas_dim/2) * math.log(2*math.pi) - operator.meas_dim * math.log(operator.sigma)
    return log_likes

def seperate_prior_pydream_sample(workdir, prior, operator, measurement, nchains=5, its=10000, nsamples=1000, nsamples_for_weight=1000000):
    log_likes = torch.zeros(len(prior.distributions), nsamples_for_weight)
    for idx, distribution in enumerate(prior.distributions):
        log_likes[idx, :] = get_log_likes(distribution, operator, measurement, nsamples_for_weight)
    fix_norm = (-1. * torch.max(log_likes).item()) 
    likes = torch.mean(torch.exp(log_likes + fix_norm), dim=-1)
    new_weights = np.array([prior.weights[idx]*likes[idx].item() for idx in range(len(prior.distributions))])
    new_weights = new_weights / np.sum(new_weights)
    samples_per_dist = np.round(nsamples * new_weights).astype(dtype=int)
    logging.info("Samples per dist were {:d}, {:d}, and {:d}".format(samples_per_dist[0], samples_per_dist[1], samples_per_dist[2]))
    likelihood = Likelihood(operator, measurement)
    tot_samples = []
    for idx, distribution in enumerate(prior.distributions):
        if samples_per_dist[idx] == 0:
            continue
        def Latin_hypercube(minn, maxn, N):
            y = np.random.rand(N, len(minn))
            x = np.zeros((N, len(minn)))
            
            for j in range(len(minn)):
                idx = np.random.permutation(N)
                P = (idx - y[:,j])/N
                x[:,j] = minn[j] + P * (maxn[j] - minn[j])
            
            return x
        history = Latin_hypercube(np.linspace(-10, -10, num=10), np.linspace(10, 10, num=10), 10000)
        np.save(os.path.join(workdir, 'history_file.npy'), history)
        start = [history[chain] for chain in range(nchains)]
        parameter = Parameter(distribution, operator.data_dim)
        likelihood = Likelihood(operator, measurement)
        samples, log_ps = core.run_dream([parameter], likelihood, nchains=nchains, multitry=False, parallel=False, niterations=its, save_history=False, 
                                         verbose=False, start=start, start_random=False, history_file = os.path.join(workdir, 'history_file.npy'))
        logging.info("The minimum and maximum log_p of the {:d}th distribution (from one chain, discarding burn in) is {:.6f}, {:.6f}".format(idx, np.min(log_ps[0][its//2:]), np.max(log_ps[0][its//2:])))
        samples_extracted = extract_samples_from_chain(samples, samples_per_dist[idx])
        tot_samples.append(samples_extracted)
    tot_samples = torch.concatenate(tot_samples)
    return tot_samples[torch.randperm(tot_samples.size()[0])]

def run_mixture_sampling(args, config, resultsfolder):
    """"
    Inputs
    args (ml_collections.ConfigDict): arguments for overall experiments
    config (list of ml_collections.ConfigDict): list of config files for running gipsda 
    workdir (str): file to save results in 
    """

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # create directory for results
    os.mkdir(resultsfolder)
    workdir = os.path.join(resultsfolder, args.name)
    os.mkdir(workdir)

    # initialize logger
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(workdir, ".log")), #TODO: change
        logging.StreamHandler()
    ]
    )

    # load deep energy based model from specified checkpoint
    with open(os.path.join(args.model_dir, 'config.pickle'), 'rb') as handle:
        model_config = pickle.load(handle)
    if args.mode != 'GT':
        model_device = model_config.device
        model = score_model.create_model(model_config)
        ema = ExponentialMovingAverage(model.parameters(), decay=model_config.model.ema_rate)
        optimizer = score_model.get_optimizer(model_config, model.parameters())
        state = dict(optimizer=optimizer, model=model, ema=ema, step=0)
        restore_path = os.path.join(args.model_dir, "checkpoint.psth")
        state = restore_checkpoint(restore_path, state, model_config.device)
        ema.copy_to(model.parameters())
        model.eval()

    if args.use_exact_score and args.mode == 'GT':
        device = torch.device('cpu')
    elif args.use_exact_score:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
         device = model_device
    logging.info("The device used is " + str(device))

    # load prior distribution 
    prior = distributions.create_distribution(model_config)
    if str(device) != 'cpu':
        prior = mix_to_device(prior, device)
    
  
    # get forward operator
    operator = forward_operators.get_operator(**config.operator)

    # get samples from joint distribution of prior and measurements
    joint_samples_file = os.path.join(resultsfolder, 'joint_samples.pickle')
    if os.path.exists(joint_samples_file): # load samples if they have already been generated
        logging.info("Pre-computed samples from joint distribution found. Loading ...")
        with open(joint_samples_file, 'rb') as handle:
            joint_samples = pickle.load(handle)
        if max(args.trial_nums) < len(joint_samples):
            num_gen_joint = 0
        else:
            logging.info("Only {:d} samples from joint distribution in file, but requested trial with sample number {:d}. Generating more samples...".format(len(joint_samples), max(args.trial_nums) + 1))
            num_gen_joint = max(args.trial_nums) + 1 - len(joint_samples)
            torch.manual_seed(num_gen_joint*1000) # needed so we dont get the same samples again due to fixed random seed
    else:
        logging.info("Pre-computed samples from joint distribution not found. Generating ...")
        num_gen_joint = max(args.trial_nums) + 1
        joint_samples = []
    for trial in range(num_gen_joint):
        trial_samples = {}
        trial_samples['sample'] =  sample = prior.sample().cpu()
        trial_samples["measurement"] = operator.measure(sample)
        joint_samples.append(trial_samples)
    with open(joint_samples_file, "wb") as handle:
        pickle.dump(joint_samples, handle)
    logging.info("Joint Samples Ready! Looping through trials ...")

    # loop through trials 
    all_results = []
    for trial in args.trial_nums:
        logging.info("Running the {:d}th trial ...".format(trial))
        trial_results = {}
        # obtain sample from prior
        trial_results["sample"] = sample = joint_samples[trial]["sample"]
        # obtain sample from likelihood
        trial_results["measurement"] = measurement = joint_samples[trial]['measurement']
        if args.mode == 'GT':    # if computing ground truth samples
            if config.operator.name == 'inpainting1D': # posterior available 
                posterior = get_mixture_gaussian_posterior(prior, torch.squeeze(measurement), operator.A, operator.sigma) # posterior on CPU
                trial_results['gt_samples'] = posterior.sample(num_samples=args.samples_per_trial) # This should be done on the CPU
                trial_results['gt_samples2'] = posterior.sample(num_samples=args.samples_per_trial)  # used in evaluation
            elif config.gt_sampling.use_partioning:
                logging.info("Running PyDream (with prior partioning) for first of two i.i.d. sample sets ....")
                trial_results['gt_samples'] = seperate_prior_pydream_sample(workdir, prior, operator, measurement, nchains=config.gt_sampling.num_chains, its=config.gt_sampling.pydream_its, 
                                                                            nsamples=args.samples_per_trial)
                logging.info("Running PyDream (with prior partioning) for second of two i.i.d. sample sets ....")
                trial_results['gt_samples2'] = seperate_prior_pydream_sample(workdir, prior, operator, measurement, nchains=config.gt_sampling.num_chains, its=config.gt_sampling.pydream_its, nsamples=args.samples_per_trial)
            else:
                logging.info("Running PyDream for first of two i.i.d. sample sets ....")
                trial_results['gt_samples'] = pydream_sample(workdir, prior, operator, measurement, nchains=config.gt_sampling.num_chains, its=config.gt_sampling.pydream_its, nsamples=args.samples_per_trial, 
                                                             seed_history=config.gt_sampling.seed_history, is_gaussian=config.gt_sampling.is_gaussian)
                logging.info("Running PyDream for second of two i.i.d. sample sets ....")
                trial_results['gt_samples2'] = pydream_sample(workdir, prior, operator, measurement, nchains=config.gt_sampling.num_chains, its=config.gt_sampling.pydream_its, nsamples=args.samples_per_trial, 
                                                              seed_history=config.gt_sampling.seed_history, is_gaussian=config.gt_sampling.is_gaussian)
        else: # if computing BIPSDA samples
            class BIPSDAModelWrapper:
                def __init__(self, model, gt_dist=None, use_exact=False):
                    self.model = model
                    self.score_model_fn = score_model.get_model_fn(model, train=False, energy=True)
                    self.score_model_fn_grad = score_model.get_model_fn(model, train=False, energy=True, use_grad=True)
                    self.dist = gt_dist
                    self.use_exact = use_exact
                def score(self, x, sigma):
                    if self.use_exact:
                        return self.score_exact(x, sigma)
                    else:
                        labels = sigma * torch.ones(x.shape[0], device=x.device)
                        return self.score_model_fn(x, labels)
                def log_p(self, x, sigma):
                    self.model.eval()
                    labels = sigma * torch.ones(x.shape[0], device=x.device)
                    return -1. * self.model(x, labels)
                def score_wgrad(self, x, sigma):
                    if self.use_exact:
                        return self.score_exact(x, sigma)
                    else:
                        labels = sigma * torch.ones(x.shape[0], device=x.device)
                        return self.score_model_fn_grad(x, labels)
                def score_exact(self, x, sigma):
                    if self.dist is None:
                        raise Exception("Exact score not implemented for this problem!")
                    convolved_dist = get_convolved_distribution(self.dist, sigma)
                    return convolved_dist.score_function(x)
            bipsda_model = BIPSDAModelWrapper(model, prior, use_exact=args.use_exact_score)
            solver = get_solver(**config.sampler, operator_name = config.operator.name, measurement_config=config.data_consist, mode=args.mode, cov_type=config.cov_type, pred_alg=config.pred_alg)
            if args.use_exact_score and config.operator.name == 'inpainting1D':
                # sample from the ground truth posterior
                posterior = get_mixture_gaussian_posterior(prior, torch.squeeze(measurement), operator.A, operator.sigma) # posterior on CPU
                x_start = posterior.sample(num_samples=args.samples_per_trial).to(device)
                # then add noise
                top_noise = solver.annealing_scheduler.sigma_max
                x_start = x_start + top_noise * torch.randn_like(x_start)
            else:
                x_start = solver.get_start(sample.repeat(args.samples_per_trial, 1).to(device))
            measurement_repeated = measurement.repeat(args.samples_per_trial, 1).to(device)
            try:
                trial_results["samples"] = solver.solve(bipsda_model, x_start, operator, measurement_repeated, record=False, verbose=args.verbose, gt=None).cpu()
                trial_results["failed_indices"] = solver.failed_indices
            except:
                logging.info("Oh no! Trial failed")
                trial_results["samples"] = None
                trial_results['failed_indices'] = None
        all_results.append(trial_results)
    logging.info("Finished Running Trials! Saving results ...")
    # save results
    with open(os.path.join(workdir, 'sample_results.pickle'), "wb") as handle:
        pickle.dump(all_results, handle)
    with open(os.path.join(workdir, 'config.pickle'), "wb") as handle:
        pickle.dump(config, handle)
    with open(os.path.join(workdir, 'args.pickle'), "wb") as handle:
        pickle.dump(args, handle)
    logging.info("Results saved! Finished")

