import ml_collections
import ml_collections.config_dict

def get_config(mode):

    config = ml_collections.ConfigDict()
    # specify model
    config.model = model = ml_collections.ConfigDict()
    model.name = 'sdm'
    model.model_config = model_config = ml_collections.ConfigDict()
    model_config.model_path = 'checkpoints/GaussMixture/Energy/checkpoint.psth'
    # specify data
    config.data = ml_collections.ConfigDict()
    # specify operator information
    config.operator = operator = ml_collections.ConfigDict()
    operator.name = 'inpainting1D'
    operator.observed_indices = [0, 1, 2, 4, 5, 6, 7, 9] 
    operator.sigma = 5.
    operator.data_dim = 10
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
    if mode == 'Lang':
        data_consist.num_steps = 100
        data_consist.lr = 5e-5
        data_consist.lr_min_ratio = 1.
    elif mode == 'MAP':
        data_consist.lambda_ = 1.
    elif mode == 'RTO':
        pass
    config.cov_type = 'identity'
    config.pred_alg = 'euler' # 'euler' or 'recursive_tweedie'

    return config