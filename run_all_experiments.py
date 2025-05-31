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


# fixedarguments
num_trials = 100
seed = 0
verbose = False
use_exact_score = False
results_folder_base = 'results/'
os.makedirs(results_folder_base, exist_ok=True)

for exper_type in ['inpainting_highnoise', 'inpainting_lownoise', 'xray_tomography', 'phase_retrieval']:
    parentresultsfolder = os.path.join(results_folder_base, exper_type)
    os.makedirs(parentresultsfolder, exist_ok=True)
    for bipsda_sampler in ['Lang', 'MAP','RTO']:
        for denoising_model in ['TU', 'ODE']:
            if (exper_type == 'inpainting_highnoise') and ((bipsda_sampler != 'RTO') or (denoising_model == 'TU')):
                continue
            exper_name = bipsda_sampler + '+' + denoising_model
            args = ml_collections.ConfigDict()
            args.name = exper_name
            args.trial_nums = [idx for idx in range(num_trials)]
            args.samples_per_trial = 10000
            args.model_dir = 'checkpoints/GaussMixture/Energy'
            args.seed = seed
            args.verbose = verbose
            args.use_exact_score = use_exact_score
            args.mode = bipsda_sampler
            if exper_type == 'xray_tomography':
                args.task = 'xray_tomography'
                config = xray_config(bipsda_sampler)
            elif exper_type == 'phase_retrieval':
                args.task = 'phase_retrieval'
                config = phase_config(bipsda_sampler)
            elif exper_type == 'inpainting_highnoise' or exper_type == 'inpainting_lownoise':
                args.task = 'inpainting'
                config = inpainting_config(bipsda_sampler)
                if exper_type == 'inpainting_highnoise':
                    config.operator.sigma = 5.
                else:
                    config.operator.sigma = .1
            else:
                raise Exception("Unknown Experiment Type must provide --exper_type = {xray_tomography, phase_retrieval, inpainting_highnoise, inpainting_lownoise}")
            if denoising_model == 'TU':
                config.sampler.diffusion_scheduler_config.num_steps = 1
                config.cov_type = 'identity'
            elif denoising_model == 'TC':
                config.sampler.diffusion_scheduler_config.num_steps = 1
                config.cov_type = 'exact'
            elif denoising_model == 'ODE':
                config.sampler.diffusion_scheduler_config.num_steps = 5
                config.cov_type = 'identity'
            else:
                raise Exception("Unknown denoising model type provided")

            # raise exception if use_exact_score is False and TC 
            if (not args.use_exact_score) and denoising_model == 'TC':
                raise Exception("Stable BIPSDA implementation that can facilitate use of TC in learned score regime not implemented yet!")
            # run
            run_mixture_sampling(args, config, parentresultsfolder)