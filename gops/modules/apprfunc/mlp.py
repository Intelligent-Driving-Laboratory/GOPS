#   Copyright (c) 2020 ocp-tools Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Author: Sun Hao

__all__=['Actor','CriticQ','QValue']


import numpy as np
import torch
import torch.nn as nn
from modules.utils.utils import get_activation_func

# import tensorboardX
# import tensorboard

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class Actor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']
        act_limit = kwargs['action_high_limit']

        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, get_activation_func(kwargs['hidden_activation']), get_activation_func(kwargs['output_activation']))
        self.act_limit =   torch.from_numpy(act_limit)

    def forward(self, obs):
        return self.act_limit * self.pi(obs)


class QValue(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim  = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_dim], nn.ReLU)

    def forward(self, obs):
        return self.q(obs)


class CriticQ(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], get_activation_func(kwargs['hidden_activation']))

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


class CriticV(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        hidden_sizes = kwargs['hidden_sizes']
        self.v = mlp([obs_dim] + list(hidden_sizes) + [1], get_activation_func(kwargs['hidden_activation']), )

    def forward(self, obs):
        return self.v(obs)

#
