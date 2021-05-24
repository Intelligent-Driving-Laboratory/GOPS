#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yao MU
#  Description: Structural definition for approximation function
#
#  Update Date: 2021-05-21, Shengbo Li: revise headline


__all__ = ['DetermPolicy', 'StochaPolicy', 'ActionValue', 'ActionValueDis', 'StateValue']

import numpy as np  # Matrix computation library
import torch
import torch.nn as nn
from modules.utils.utils import get_activation_func


# Define MLP function
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


# Count parameter number of MLP
def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


# Deterministic policy
class DetermPolicy(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']
        action_high_limit = kwargs['action_high_limit']
        action_low_limit = kwargs['action_low_limit']

        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes,
                      get_activation_func(kwargs['hidden_activation']),
                      get_activation_func(kwargs['output_activation']))
        self.action_high_limit = torch.from_numpy(action_high_limit)
        self.action_low_limit = torch.from_numpy(action_low_limit)

    def forward(self, obs):
        action = (self.action_high_limit-self.action_low_limit)/2 * torch.tanh(self.pi(obs))\
                 + (self.action_high_limit + self.action_low_limit)/2
        return action


# Stochastic Policy
class StochaPolicy(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']
        action_high_limit = kwargs['action_high_limit']
        action_low_limit = kwargs['action_low_limit']

        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.mean = mlp(pi_sizes,
                        get_activation_func(kwargs['hidden_activation']),
                        get_activation_func(kwargs['output_activation']))
        self.std = mlp(pi_sizes,
                       get_activation_func(kwargs['hidden_activation']),
                       get_activation_func(kwargs['output_activation']))
        self.action_high_limit = torch.from_numpy(action_high_limit)
        self.action_low_limit = torch.from_numpy(action_low_limit)

    def forward(self, obs):
        action_mean = (self.action_high_limit - self.action_low_limit) / 2 * torch.tanh(self.mean(obs)) \
                 + (self.action_high_limit + self.action_low_limit) / 2
        return action_mean, torch.exp(self.std(obs))


class ActionValue(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1],
                     get_activation_func(kwargs['hidden_activation']),
                     get_activation_func(kwargs['output_activation']))

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


class ActionValueDis(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']
        print(kwargs['output_activation'])
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_dim],
                     get_activation_func(kwargs['hidden_activation']),
                     get_activation_func(kwargs['output_activation']))

    def forward(self, obs):
        return self.q(obs)


class StateValue(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        hidden_sizes = kwargs['hidden_sizes']
        self.v = mlp([obs_dim] + list(hidden_sizes) + [1],
                     get_activation_func(kwargs['hidden_activation']),
                     get_activation_func(kwargs['output_activation']))

    def forward(self, obs):
        return self.v(obs)

#
