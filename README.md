# BIPSDA

Code for "Bayesian Inverse Problem Solvers through Diffusion Annealing" (BIPSDA), a framework for solving Bayesian inverse problems using diffusion models.

The BIPDSA framework was introduced in ``Can Diffusion Models Provide Rigorous Uncertainty Quantification for Bayesian Inverse Problems?'' The published paper can be found [here](10.1109/OJSP.2025.3597867). The results of the numerical experiments conducted for this paper can be accessed directly [here](https://doi.org/10.7910/DVN/0L5KGB) to enable the analysis of the approximate posterior samples produced by the BIPDSA algorithms without running the code. 

# Installation

It is recommended to use the [Code Ocean version of this repository](https://codeocean.com/capsule/6733743/tree), as this version contains the trained diffusion model weights and is designed to be run without any modification of the code. 

If you would like to use the Github version instead, note that the codebase has minimal dependencies and is designed to be easy to install. Simply clone this repository and setup a virtual environment or conda environment 
with PyTorch, ml_collections, logging, and matplotlib installed. 

# Use 

There are two main scripts in this repository: (1) experiment_runner.py, which runs BISPDA experiments, and (2) eval_runner.py, which evaluates the result 
of a BIPSDA experiment. 

To use experiment_runner, simply run "python3 experiment_runner.py" with the appropriate flags to specify the type of experiment, number of trials, 
and other experimental details. Here is a complete list of options. Only the experiment type is required; all other options have defaults. 
1. --exper_type (string): Name of experiment type - phase_retrieval, inpainting_highnoise, inpainting_lownoise, or xray_tomography (required option)
2. --parentresultsfolder (string): directory within which results folder will be created (default=;results/Experiment') 
3. -exper_name (string): Name of experiment (default=time.ctime())
4. --seed (int): 'Random number seed (default = 0)'
5. --verbose (boolean flag): Verbose flag (default=False)'
6. --num_trials (int): Number of trials to run (default = 1)
7. --bipsda_sampler (string): Sampler type - Lang, MAP, or RTO, or GT (default='Lang')
8. --denoising_model (string):' Model of denoising distribution - TC, TU, or ODE (default='TU')
9. --use_exact_score (boolean flag): Whether to use analytic score, as opposed to learned score (default=False)'

After running an experiment, you can evaluate the results (which includes both plotting and quantitative evaluation) by running
"python3 eval_runner.py". Here relative paths to the folder with the results of the experiment, as well as a folder with the reference
ground-truth samples, must be passed. In other words, with the same choice of parentresultsfolder, experiment_runner must be 
run in both 'GT' mode and with one of the other BIPSDA sampler options to evaluate the performance of the BIPSDA sampler. The full list of options 
for the eval runner are as follows:
1. --exper_type (string): Name of experiment type - phase_retrieval, inpainting_highnoise, inpainting_lownoise, or xray_tomography (required option)
2. --sampledir (string): Relative path to directory where samples from the numerical experiment were stored (required option)
3. --gtdir (string): Relative path to directory where ground-truth samples from the numerical experiment of interest were stored (reqired option)
4. --eval_name (string): Name to use for folder within sampledir where results are stored (default='Eval:Default')
5. --plot_options (string): Plotting mode - first, none, all, or first10 (default='first10')
