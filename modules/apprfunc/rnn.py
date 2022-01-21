#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yao MU
#  Description: Structural definition for approximation function
#
#  Update Date: 2021-05-21, Shengbo Li: revise headline

__all__=['DetermPolicy','StochaPolicy','ActionValue','ActionValueDis','StateValue']


import numpy as np
import torch
import torch.nn as nn
from modules.utils.utils import get_activation_func


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class DetermPolicy(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim'][1]
        act_dim = kwargs['act_dim']
        action_high_limit = kwargs['act_high_lim']
        action_low_limit = kwargs['act_low_lim']
        hidden_sizes = kwargs['hidden_sizes']
        pi_sizes = list(hidden_sizes) + [act_dim]
        self.rnn = nn.RNN(obs_dim, hidden_sizes[0], 1, batch_first=True)
        self.pi = mlp(pi_sizes, get_activation_func(kwargs['hidden_activation']), get_activation_func(kwargs['output_activation']))
        self.register_buffer('act_high_lim', torch.from_numpy(action_high_limit))
        self.register_buffer('act_low_lim', torch.from_numpy(action_low_limit))

    def forward(self, obs):
        _, h = self.rnn(obs)
        action = (self.act_high_lim-self.act_low_lim)/2 * torch.tanh(self.pi(h.squeeze(0)))\
                 + (self.act_high_lim + self.act_low_lim)/2
        return action


class StochaPolicy(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim'][1]
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']
        action_high_limit = kwargs['act_high_lim']
        action_low_limit = kwargs['act_low_lim']
        self.min_log_std = kwargs['min_log_std']
        self.max_log_std = kwargs['max_log_std']

        pi_sizes = list(hidden_sizes) + [act_dim]
        self.rnn = nn.RNN(obs_dim, hidden_sizes[0], 1, batch_first=True)
        self.mean = mlp(pi_sizes,
                        get_activation_func(kwargs['hidden_activation']),
                        get_activation_func(kwargs['output_activation']))
        self.log_std = mlp(pi_sizes,
                           get_activation_func(kwargs['hidden_activation']),
                           get_activation_func(kwargs['output_activation']))
        self.register_buffer('act_high_lim', torch.from_numpy(action_high_limit))
        self.register_buffer('act_low_lim', torch.from_numpy(action_low_limit))

    def forward(self, obs):
        _, h = self.rnn(obs)
        h = h.squeeze(0)
        action_mean = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(self.mean(h)) \
                      + (self.act_high_lim + self.act_low_lim) / 2
        action_std = torch.clamp(self.log_std(h), self.min_log_std, self.max_log_std).exp()
        return torch.cat((action_mean, action_std), dim=-1)


class ActionValue(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim'][1]
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']
        self.rnn = nn.RNN(obs_dim, hidden_sizes[0], 1, batch_first=True)
        self.q = mlp(list([hidden_sizes[0]+act_dim]) + list(hidden_sizes[1:]) + [1], get_activation_func(kwargs['hidden_activation']))

    def forward(self, obs, act):
        _, h = self.rnn(obs)
        input = torch.cat([h.squeeze(0), act], dim=-1)
        q = self.q(input)
        return torch.squeeze(q, -1)


class ActionValueDis(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim'][1]
        act_num = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']
        self.rnn = nn.RNN(obs_dim, hidden_sizes[0], 1, batch_first=True)
        self.q = mlp(list(hidden_sizes) + [act_num],
                     get_activation_func(kwargs['hidden_activation']),
                     get_activation_func(kwargs['output_activation']))

    def forward(self, obs):
        _, h = self.rnn(obs)
        return self.q(h.squeeze(0))


class StateValue(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim'][1]
        hidden_sizes = kwargs['hidden_sizes']
        self.rnn = nn.RNN(obs_dim, hidden_sizes[0], 1, batch_first=True)
        self.v = mlp(list(hidden_sizes) + [1],
                     get_activation_func(kwargs['hidden_activation']),
                     get_activation_func(kwargs['output_activation']))

    def forward(self, obs):
        _, h = self.rnn(obs)
        v = self.v(h.squeeze(0))
        return torch.squeeze(v, -1)
