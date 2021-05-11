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


class RBF(nn.Module):
    def __init__(self, input_dim=5,out_dim=10, kernel_num=3):
        self.input_dim=input_dim
        self.kernel = kernel_num
        self.C = nn.Parameter(torch.randn(1, self.kernel, input_dim))  # (n,k,5)
        self.sigma = nn.Parameter(torch.randn(1, self.kernel))  # (n,k)
        self.w = nn.Parameter(torch.randn(1, out_dim, self.kernel))  # (n,1,k)
        self.b = nn.Parameter(torch.randn(1, out_dim, 1))  # (n,1,1)
        self.tanh = nn.Tanh()

    def forward(self, x):  # (n,5)
        r = torch.sqrt(torch.sum((x.view(-1, 1, self.input_dim) - self.C) ** 2, dim=-1))  # (n,k)
        phi = torch.exp(-r ** 2 / (2 * self.sigma ** 2)).unsqueeze(-1)  # (n,k,1)
        return self.tanh(self.w @ phi + self.b).squeeze(-1)

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
        act_limit = kwargs['action_high_limit']
        #num_kernel = kwargs['num_rbf_kernel']
        self.pi = RBF(obs_dim,act_dim)
        self.act_limit =   torch.from_numpy(act_limit)

    def forward(self, obs):
        return self.act_limit * self.pi.forward(obs)

class StochaPolicy(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        act_limit = kwargs['action_high_limit']
        #num_kernel = kwargs['num_rbf_kernel']

        self.mean = RBF(obs_dim, act_dim)
        self.std = RBF(obs_dim, act_dim)
        self.act_limit =   torch.from_numpy(act_limit)

    def forward(self, obs):
        return self.act_limit * self.mean(obs), torch.exp(self.std(obs))


class ActionValue(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        self.q = RBF(obs_dim + act_dim, 1)


    def forward(self, obs, act):
        q = self.q.forward(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


class ActionValueDis(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim  = kwargs['obs_dim']
        #num_kernel = kwargs['num_rbf_kernel']
        self.q = RBF(obs_dim, 1)

    def forward(self, obs):
        return  self.q.forward(obs)




class StateValue(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        self.v = RBF(obs_dim, 1)

    def forward(self, obs):
        return self.v.forward(obs)