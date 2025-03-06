""""
Various functions and classes required to form and use score model
"""

import torch
import torch.nn as nn
from __future__ import division
from __future__ import unicode_literals
import torch
import os
import logging
import torch.optim as optim
import numpy as np

import sde_lib

_MODELS = {}

def register_model(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_model(name):
  return _MODELS[name]


def create_model(config):
  """Create the score model."""
  model_name = config.model.name
  score_model = get_model(model_name)(config)
  score_model = score_model.to(config.device)
  return score_model

@register_model(name='ToyConditionalModel')
class ToyConditionalModel(nn.Module):

    def __init__(self, config):
        super(ToyConditionalModel, self).__init__()
        self.dim = dim = config.data.dim
        hidden_dim = config.model.hidden_dim
        if config.model.activation == 'ReLU':
            self.activation = nn.ReLU()
        elif config.model.activation == 'Softplus':
            self.activation = nn.Softplus()
        else:
            raise Exception("Unknown activation function type")
        self.lin_in = nn.Linear(dim + 1, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.inner_layers1 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(config.model.num_hidden1)])
        self.middle_layer = nn.Linear(hidden_dim + 1, hidden_dim)
        self.inner_layers2 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(config.model.num_hidden2)])
        #self.energy_flag = config.model.is_energy
        if config.model.is_energy == False:
          self.lin_out = nn.Linear(hidden_dim + 1, dim)
        else:
          self.lin_out = nn.Linear(hidden_dim + 1, 1)
        #if self.energy_flag:
        #   self.lin_final = nn.Linear(dim, 1)
        if hasattr(config.model, 'use_batch_norm'):
           self.use_batch_norm = config.model.use_batch_norm
        else:
           self.use_batch_norm = True

    def forward(self, x, noise_labels):
        noise_labels = noise_labels[:, None] 
        x = torch.cat((x, noise_labels), dim=1)
        x = self.lin_in(x)
        x = self.activation(x)
        if self.use_batch_norm:
          x = self.batch_norm(x)
        for layer in self.inner_layers1:
            x = layer(x)
            x = self.activation(x)
        x = torch.cat((x, noise_labels), dim=1)
        x = self.middle_layer(x)
        x = torch.mul(self.activation(x), x)
        for layer in self.inner_layers2:
            x = layer(x)
            x = self.activation(x)
        x = torch.cat((x, noise_labels), dim=1)
        x = self.lin_out(x)
        # if self.energy_flag:
        #    x = self.activation(x)
        #    x = self.lin_final(x)
        return x

# Ported from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.p
# Modified from https://raw.githubusercontent.com/fadel/pytorch_ema/master/torch_ema/ema.py
# Partially based on: https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py
class ExponentialMovingAverage:
  """
  Maintains (exponential) moving average of a set of parameters.
  """

  def __init__(self, parameters, decay, use_num_updates=True):
    """
    Args:
      parameters: Iterable of `torch.nn.Parameter`; usually the result of
        `model.parameters()`.
      decay: The exponential decay.
      use_num_updates: Whether to use number of updates when computing
        averages.
    """
    if decay < 0.0 or decay > 1.0:
      raise ValueError('Decay must be between 0 and 1')
    self.decay = decay
    self.num_updates = 0 if use_num_updates else None
    self.shadow_params = [p.clone().detach()
                          for p in parameters if p.requires_grad]
    self.collected_params = []

  def update(self, parameters):
    """
    Update currently maintained parameters.

    Call this every time the parameters are updated, such as the result of
    the `optimizer.step()` call.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; usually the same set of
        parameters used to initialize this object.
    """
    decay = self.decay
    if self.num_updates is not None:
      self.num_updates += 1
      decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
    one_minus_decay = 1.0 - decay
    with torch.no_grad():
      parameters = [p for p in parameters if p.requires_grad]
      for s_param, param in zip(self.shadow_params, parameters):
        s_param.sub_(one_minus_decay * (s_param - param))

  def copy_to(self, parameters):
    """
    Copy current parameters into given collection of parameters.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        updated with the stored moving averages.
    """
    parameters = [p for p in parameters if p.requires_grad]
    for s_param, param in zip(self.shadow_params, parameters):
      if param.requires_grad:
        param.data.copy_(s_param.data)

  def store(self, parameters):
    """
    Save the current parameters for restoring later.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        temporarily stored.
    """
    self.collected_params = [param.clone() for param in parameters]

  def restore(self, parameters):
    """
    Restore the parameters stored with the `store` method.
    Useful to validate the model with EMA parameters without affecting the
    original optimization process. Store the parameters before the
    `copy_to` method. After validation (or model saving), use this to
    restore the former parameters.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        updated with the stored parameters.
    """
    for c_param, param in zip(self.collected_params, parameters):
      param.data.copy_(c_param.data)

  def state_dict(self):
    return dict(decay=self.decay, num_updates=self.num_updates,
                shadow_params=self.shadow_params)

  def load_state_dict(self, state_dict):
    self.decay = state_dict['decay']
    self.num_updates = state_dict['num_updates']
    self.shadow_params = state_dict['shadow_params']


def restore_checkpoint(ckpt_dir, state, device):
  if not os.path.isdir(ckpt_dir):
    os.mkdir(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir)


  def get_optimizer(config, params):
    """Returns a flax optimizer object based on `config`."""
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                            weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
        f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()


def create_model(config):
    """Create the score model."""
    model_name = config.model.name
    score_model = get_model(model_name)(config)
    score_model = score_model.to(config.device)
    score_model = torch.nn.DataParallel(score_model)
    return score_model


def get_model_fn(model, train=False, energy=False, use_grad=False):
  """Create a function to give the output of the score-based model.

  Args:
    model: The score model.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function.
  """

  def model_fn(x, labels):
    """Compute the output of the score-based model.

    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.

    Returns:
      A tuple of (model output, new mutable states)
    """

    if train:
      model.train()
    else:
      model.eval()

    if not energy:
      if (not train) and (not use_grad):
        with torch.no_grad():
          return model(x, labels)
      else:
        return model(x, labels)
    else:
      x.requires_grad_(True)
      logp = -model(x, labels).sum()
      grad = autograd.grad(logp, x, create_graph=True)[0]
      if (not train) and not(use_grad):
        grad = grad.detach()
      if not(use_grad):
        x.requires_grad_(False)
      return grad

  return model_fn


def get_score_fn(sde, model, train=False, continuous=False, energy=False, use_grad=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.

  Returns:
    A score function.
  """
  model_fn = get_model_fn(model, train=train, energy=energy, use_grad=use_grad)

  if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
    def score_fn(x, t):
      # Scale neural network output by standard deviation and flip sign
      if continuous or isinstance(sde, sde_lib.subVPSDE):
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        labels = t * 999
        score = model_fn(x, labels)
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        labels = t * (sde.N - 1)
        score = model_fn(x, labels)
        std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

      score = -score / std[:, None]
      return score

  elif isinstance(sde, sde_lib.VESDE):
    def score_fn(x, t):
      if continuous:
        labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VE-trained models, t=0 corresponds to the highest noise level
        labels = sde.T - t
        labels *= sde.N - 1
        labels = torch.round(labels).long()

      score = model_fn(x, labels)
      return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn
