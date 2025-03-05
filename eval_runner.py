"""
Runner for evaluating results of experiment
"""

import argparse

import ml_collections

from evaluate_results import run_eval

parser = argparse.ArgumentParser()

parser.add_argument('-s', '--exper_type', type=str, help='Name of experiment type (phase_retrieval, inpainting, or xray_tomography)')
parser.add_argument('-s', '--exper_type', type=str, help='Name of experiment type (phase_retrieval, inpainting, or xray_tomography)')
parser.add_argument('-s', '--eval_name', type=str, help='Name for evaluation run')
parser.add_argument('-s', '--plot_options', type=str, help='Plotting mode (first, none, all, or first10)')
parser.add_argument('-f', '--flag', action='store_true', help='A boolean flag')


sampledir = 'results/MixtureDistribution/020325SystemicXRayExperiments/Mode_cheatRTO_UseExact_True_UseTweedie_cov+mean/'
gtdir =  'results/MixtureDistribution/020325SystemicXRayExperiments/gt_samples/'
eval_name = 'Eval:Default'

eval_ops = ml_collections.ConfigDict()
eval_ops.plot_options = 'first10'
eval_ops.num_plot = 1000
eval_ops.seed = 0
if exper_type == 'phase_retrieval':
    eval_ops.mmd_kernel_bandwidth = 12.6182
    eval_ops.cmd_decay_rate = 16.9327
elif exper_type == 'inpainting':
    eval_ops.mmd_kernel_bandwidth = 4.9753 
    eval_ops.cmd_decay_rate = 6.3731
elif exper_type == 'xray_tomography':
    eval_ops.mmd_kernel_bandwidth = 4.3266
    eval_ops.cmd_decay_rate = 4.3931
else:
    raise Exception("Unknown Experiment Type")
run_eval(eval_ops, sampledir, gtdir, eval_name)