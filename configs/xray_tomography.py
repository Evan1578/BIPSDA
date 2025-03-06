import ml_collections
import ml_collections.config_dict
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
    operator.name = 'xray_tomography'
    dims = (15, 10)
    operator_seed = 0
    generator = torch.Generator()
    generator.manual_seed(operator_seed)
    operator.I0 = 1000
    operator.A = .01 + .04*torch.rand(dims, generator=generator)
    # specify GT sampling approach
    config.gt_sampling = gt_sampling = ml_collections.ConfigDict()
    gt_sampling.num_chains = 10
    gt_sampling.pydream_its = 200000
    gt_sampling.use_partioning = False
    gt_sampling.is_gaussian = False
    gt_sampling.seed_history = False
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
        data_consist.num_steps = 1000
        data_consist.lr = 5e-5
        data_consist.lr_min_ratio = 1.
    elif (mode == 'map_estimation') or (mode == 'RTO'):
        data_consist.lambda_ = 1
        data_consist.max_its = 40
        data_consist.solver = 'lbfgs'
        data_consist.solver_params = {'lr': 1, 'line_search_fn':'strong_wolfe'}
    config.cov_type = 'identity'
    config.pred_alg = 'euler' # 'euler' or 'recursive_tweedie'
    return config