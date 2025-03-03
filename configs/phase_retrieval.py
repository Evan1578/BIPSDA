import ml_collections
import ml_collections.config_dict
import yaml
import torch

def get_config(mode):
    config = ml_collections.ConfigDict()
    # specify model
    config.model = model = ml_collections.ConfigDict()
    model.name = 'sdm'
    model.model_config = model_config = ml_collections.ConfigDict()
    model_config.model_path = 'checkpoints/GaussMixture/Energy/checkpoint.psth'
    # specify data
    config.data = data = ml_collections.ConfigDict()
    # specify operator information
    config.operator = operator = ml_collections.ConfigDict()
    operator.name = 'gaussphaseretrieval1D'
    operator.dims = (5, 10)
    operator.seed = 0
    operator.sigma = 25.
    # specify GT sampling approach
    config.gt_sampling = gt_sampling = ml_collections.ConfigDict()
    gt_sampling.num_chains = 40
    gt_sampling.pydream_its = 200000
    gt_sampling.use_partioning = False
    # specify sampler (excluding data consistency) information
    sampler = ml_collections.ConfigDict()
    sampler['latent'] = False
    sampler['annealing_scheduler_config'] = annealing_schedule = ml_collections.ConfigDict()
    annealing_schedule['num_steps'] = 200
    annealing_schedule['sigma_max'] = 10
    annealing_schedule['sigma_min'] = .01
    annealing_schedule['sigma_final'] =  0
    annealing_schedule['schedule'] = 'linear'
    annealing_schedule['timestep'] = 'poly-7'
    sampler['diffusion_scheduler_config'] = diffusion_schedule = ml_collections.ConfigDict()
    diffusion_schedule['num_steps'] = 5
    diffusion_schedule['sigma_min'] = .01
    diffusion_schedule['sigma_final'] =  0
    diffusion_schedule['schedule'] = 'linear'
    diffusion_schedule['timestep'] = 'poly-7'
    config.sampler = sampler
    # specify data consistency updates
    config.data_consist = data_consist =  ml_collections.ConfigDict()
    if mode == 'langevin_dynamics':
        data_consist.num_its = 500
        data_consist.step_size = 2e-1
        data_consist.metric_type = 'gauss_newton'
    elif (mode == 'map_estimation') or (mode == 'RTO'):
        #data_consist.lambda_ = 1
        data_consist.max_its = 40
        data_consist.solver = 'lbfgs'
        data_consist.solver_params = {'lr': 1, 'line_search_fn':'strong_wolfe'}
    config.cov_type = 'identity'
    config.pred_alg = 'euler' # 'euler' or 'recursive_tweedie'
    return config