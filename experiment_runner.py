"""
Runner for all BIPSDA experiments
"""

import argparse

import ml_collections

from solve_inverse_problem import run_mixture_sampling
from configs.inpainting import get_config as inpainting_config
from configs.phase_retrieval import get_config as phase_config
from configs.xray_tomography import get_config as xray_config
import os
import time



# parser
parser = argparse.ArgumentParser()
parser.add_argument('--exper_type', type=str, help='Name of experiment type (phase_retrieval, inpainting_highnoise, inpainting_lownoise, or xray_tomography)')
parser.add_argument('--parentresultsfolder', type=str, help='Directory within which results folder will be created', default='results/Experiment/')
parser.add_argument('--exper_name', type=str, help='Name of experiment', default=time.ctime())
parser.add_argument('--seed', type=int, help='Random number seed (default = 0)', default=0)
parser.add_argument('--verbose', action='store_true', help='Verbose flag (boolean)', default=False)
parser.add_argument('--num_trials', type=int, help='Number of trials to run (default = 1)', default=1)
parser.add_argument('--bipsda_sampler', type=str, help='Sampler (Lang, MAP, or RTO, or GT)', default='Lang')
parser.add_argument('--denoising_model', type=str, help='Model of denoising distribution (TC, TU, or ODE)', default='TU')
parser.add_argument('--use_exact_score', action='store_true', help='Whether to use analytic score, as opposed to learned score (boolean)', default=False)

# parse arguments
passed_args = parser.parse_args()

# get arguments
parentresultsfolder = passed_args.parentresultsfolder
exper_type = passed_args.exper_type

args = ml_collections.ConfigDict()
args.name = passed_args.exper_name
args.trial_nums = [idx for idx in range(passed_args.num_trials)]
args.samples_per_trial = 10000
args.model_dir = 'checkpoints/GaussMixture/Energy'
args.seed = passed_args.seed
args.verbose = passed_args.verbose
args.use_exact_score = passed_args.use_exact_score
args.mode = passed_args.bipsda_sampler
if exper_type == 'xray_tomography':
    args.task = 'xray_tomography'
    config = xray_config(passed_args.bipsda_sampler)
elif exper_type == 'phase_retrieval':
    args.task = 'phase_retrieval'
    config = phase_config(passed_args.bipsda_sampler)
elif exper_type == 'inpainting_highnoise' or exper_type == 'inpainting_lownoise':
    args.task = 'inpainting'
    config = inpainting_config(passed_args.bipsda_sampler)
    if exper_type == 'inpainting_highnoise':
        config.operator.sigma = 5.
    else:
        config.operator.sigma = .1
else:
    raise Exception("Unknown Experiment Type must provide --exper_type = {xray_tomography, phase_retrieval, inpainting_highnoise, inpainting_lownoise}")
if passed_args.denoising_model == 'TU':
    config.sampler.diffusion_scheduler_config.num_steps = 1
    config.cov_type = 'identity'
elif passed_args.denoising_model == 'TC':
    config.sampler.diffusion_scheduler_config.num_steps = 1
    config.cov_type = 'exact'
elif passed_args.denoising_model == 'ODE':
    config.sampler.diffusion_scheduler_config.num_steps = 5
    config.cov_type = 'identity'
else:
    raise Exception("Unknown denoising model type provided")

# raise exception if use_exact_score is False and TC 
if (not args.use_exact_score) and passed_args.denoising_model == 'TC':
    raise Exception("Stable BIPSDA implementation that can facilitate use of TC in learned score regime not implemented yet!")

# run
run_mixture_sampling(args, config, parentresultsfolder)