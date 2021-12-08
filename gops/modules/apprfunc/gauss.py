#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yao MU
#  Description: Structural definition for approximation function
#
#  Update Date: 2021-05-21, Shengbo Li: revise headline

__all__ = ['DetermPolicy', 'StochaPolicy', 'ActionValue', 'ActionValueDis', 'StateValue']

import numpy as np
import torch
import torch.nn as nn
from modules.utils.utils import get_activation_func


class RBF(nn.Module):
    def __init__(self, input_dim, out_dim, kernel_num):
        super().__init__()
        self.input_dim = input_dim
        self.kernel = kernel_num
        self.C = nn.Parameter(torch.randn(1, self.kernel, input_dim))
        self.sigma_square = nn.Parameter(torch.abs(torch.randn(1, self.kernel))+0.1)
        self.w = nn.Parameter(torch.randn(1, out_dim, self.kernel))
        self.b = nn.Parameter(torch.randn(1, out_dim, 1))

    def forward(self, x):  # (n,5)
        r = torch.sum((x.view(-1, 1, self.input_dim) - self.C) ** 2, dim=-1)  # dim(kernel)
        phi = torch.exp(-r/ (2 * torch.abs(self.sigma_square))).unsqueeze(-1)
        return (self.w @ phi + self.b).squeeze(-1)


class DetermPolicy(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        num_kernel = kwargs['num_kernel']
        self.pi = RBF(obs_dim, act_dim, num_kernel)
        self.register_buffer('act_high_lim', torch.from_numpy(kwargs['act_high_lim']))
        self.register_buffer('act_low_lim', torch.from_numpy(kwargs['act_low_lim']))

    def forward(self, obs):
        action = (self.act_high_lim-self.act_low_lim)/2 * self.pi.forward(obs)\
                 + (self.act_high_lim + self.act_low_lim)/2
        return action


class StochaPolicy(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        num_kernel = kwargs['num_kernel']

        self.mean = RBF(obs_dim, act_dim, num_kernel)
        self.std = RBF(obs_dim, act_dim, num_kernel)
        self.min_log_std = kwargs['min_log_std']
        self.max_log_std = kwargs['max_log_std']
        self.register_buffer('act_high_lim', torch.from_numpy(kwargs['act_high_lim']))
        self.register_buffer('act_low_lim', torch.from_numpy(kwargs['act_low_lim']))

    def forward(self, obs):
        action_mean = (self.act_high_lim - self.act_low_lim) / 2 * self.mean(obs) \
                      + (self.act_high_lim + self.act_low_lim) / 2
        action_std = torch.clamp(self.std(obs), self.min_log_std, self.max_log_std).exp()
        return torch.cat((action_mean, action_std), dim=-1)


class ActionValue(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        num_kernel = kwargs['num_kernel']
        self.q = RBF(obs_dim + act_dim, 1, num_kernel)

    def forward(self, obs, act):
        q = self.q.forward(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


class ActionValueDis(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        num_kernel = kwargs['num_kernel']
        self.q = RBF(obs_dim, act_dim, num_kernel)

    def forward(self, obs):
        return self.q.forward(obs)


class StateValue(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        num_kernel = kwargs['num_kernel']
        self.v = RBF(obs_dim, 1, num_kernel)

    def forward(self, obs):
        return self.v.forward(obs)
