#   Copyright (c) 2020 ocp-tools Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Author: Sun Hao

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
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']
        act_limit = kwargs['action_high_limit']
        hidden_sizes = kwargs['hidden_sizes']
        pi_sizes = list(hidden_sizes[1:]) + [act_dim]
        self.rnn = nn.RNN(obs_dim, hidden_sizes[0], 1)
        self.pi = mlp(pi_sizes, get_activation_func(kwargs['hidden_activation']), get_activation_func(kwargs['output_activation']))
        self.act_limit =   torch.from_numpy(act_limit)

    def forward(self, obs):
        obs = obs.unsqueeze(0)
        _, h = self.rnn(obs)
        return self.act_limit * self.pi(h.squeeze(0))

class StochaPolicy(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']
        act_limit = kwargs['action_high_limit']
        hidden_sizes = kwargs['hidden_sizes']
        pi_sizes = list(hidden_sizes[1:]) + [act_dim]
        self.rnn = nn.RNN(obs_dim, hidden_sizes[0], 1)
        self.mean = mlp(pi_sizes, get_activation_func(kwargs['hidden_activation']), get_activation_func(kwargs['output_activation']))
        self.std = mlp(pi_sizes, get_activation_func(kwargs['hidden_activation']),
                        get_activation_func(kwargs['output_activation']))
        self.act_limit =   torch.from_numpy(act_limit)

    def forward(self, obs):
        obs = obs.unsqueeze(0)
        _, h = self.rnn(obs)
        return self.act_limit * self.pi(h.squeeze(0)), torch.exp(self.std(h.squeeze(0)))

class ActionValue(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']
        self.rnn = nn.RNN(obs_dim+act_dim, hidden_sizes[0],1)
        self.q = mlp(list(hidden_sizes[1:]) + [1], get_activation_func(kwargs['hidden_activation']))

    def forward(self, obs, act):
        input = torch.cat([obs, act], dim=-1)
        input = input.unsqueeze(0)
        _, h = self.rnn(input)
        q = self.q(h.squeeze(0))
        return torch.squeeze(q, -1)


class ActionValueDis(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim  = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']
        self.rnn = nn.RNN(obs_dim, hidden_sizes[0],1)
        self.q = mlp(list(hidden_sizes[1:]) + [act_dim], nn.ReLU)

    def forward(self, obs):
        obs = obs.unsqueeze(0)
        _, h = self.rnn(obs)
        return self.q(h.squeeze(0))


class StateValue(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        hidden_sizes = kwargs['hidden_sizes']
        self.rnn = nn.RNN(obs_dim, hidden_sizes[0],1)
        self.v = mlp(list(hidden_sizes[1:]) + [1], get_activation_func(kwargs['hidden_activation']))

    def forward(self, obs):
        obs = obs.unsqueeze(0)
        _, h = self.rnn(obs)
        v = self.v(h.squeeze(0))
        return torch.squeeze(v, -1)