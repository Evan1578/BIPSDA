"""
Runner for evaluating results of experiment
"""

import argparse

import ml_collections

from evaluate_results import run_eval

# parser
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--exper_type', type=str, help='Name of experiment type (phase_retrieval, inpainting_highnoise, inpainting_lownoise, or xray_tomography)')
parser.add_argument('-s', '--eval_name', type=str, help='Name for evaluation run', default='Eval:Default')
parser.add_argument('-s', '--plot_options', type=str, help='Plotting mode (first, none, all, or first10)', default='first10')
parser.add_argument('-s', '--sampledir', type=str, help='directory where samples from the numerical experiment were stored')
parser.add_argument('-s', '--gtdir', type=str, help='directory where ground-truth samples from the numerical experiment of interest were stored')

# parse arguments
args = parser.parse_args()

# get arguments
try:
    sampledir = args.sampledir
except:
    raise Exception("No Folder with experiment samples was provided")
try:
    gtdir = args.gtdir
except:
    raise Exception("No Folder with ground-truth samples was provided")
try:
    exper_type = args.exper_type
except:
    raise Exception("Must specify experiment type")

eval_name = args.eval_name
eval_ops = ml_collections.ConfigDict()
eval_ops.plot_options = args.plot_options
eval_ops.num_plot = 1000
eval_ops.seed = 0
if exper_type == 'phase_retrieval':
    eval_ops.mmd_kernel_bandwidth = 12.6182
    eval_ops.cmd_decay_rate = 16.9327
elif exper_type == 'inpainting_highnoise':
    eval_ops.mmd_kernel_bandwidth = 4.9753 
    eval_ops.cmd_decay_rate = 6.3731
elif exper_type == 'inpainting_lownoise':
    eval_ops.mmd_kernel_bandwidth = 4.9753 # TODO: fix
    eval_ops.cmd_decay_rate = 6.3731 # TODO: fix
elif exper_type == 'xray_tomography':
    eval_ops.mmd_kernel_bandwidth = 4.3266
    eval_ops.cmd_decay_rate = 4.3931
run_eval(eval_ops, sampledir, gtdir, eval_name)