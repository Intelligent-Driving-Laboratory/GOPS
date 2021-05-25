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

def conv_func(obs_dim):
    return nn.Sequential(nn.Conv2d(obs_dim[-1], 32, kernel_size=4, stride=2, padding=3),
                         nn.GELU(),
                         nn.MaxPool2d(kernel_size=2, stride=2),
                         nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2),
                         nn.GELU(),
                         nn.MaxPool2d(kernel_size=2, stride=2),
                         nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2),
                         nn.GELU())

def _get_conv_out_size(obs_dim):
        out = conv_func(torch.zeros(obs_dim).unsqueeze(0).permute(0, 3, 1, 2))
        return int(np.prod(out.size()))

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class DetermPolicy(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']
        act_limit = kwargs['action_high_limit']
        num_feature = _get_conv_out_size(obs_dim)
        pi_sizes = [num_feature] + list(hidden_sizes) + [act_dim]
        self.conv = conv_func(obs_dim)
        self.pi = mlp(pi_sizes, get_activation_func(kwargs['hidden_activation']), get_activation_func(kwargs['output_activation']))
        self.act_limit =   torch.from_numpy(act_limit)

    def forward(self, obs):
        feature = self.conv(obs)
        return self.act_limit * self.pi(feature)

class StochaPolicy(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']
        act_limit = kwargs['action_high_limit']
        num_feature = _get_conv_out_size(obs_dim)
        pi_sizes = [num_feature] + list(hidden_sizes) + [act_dim]
        self.conv = conv_func(obs_dim)
        self.pi = mlp(pi_sizes, get_activation_func(kwargs['hidden_activation']), get_activation_func(kwargs['output_activation']))
        self.act_limit =   torch.from_numpy(act_limit)

    def forward(self, obs):
        feature = self.conv(obs)
        return self.act_limit * self.pi(feature), torch.exp(self.std(feature))


class ActionValue(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']
        num_feature = _get_conv_out_size(obs_dim)
        self.q = mlp([num_feature + act_dim] + list(hidden_sizes) + [1], get_activation_func(kwargs['hidden_activation']))

    def forward(self, obs, act):
        feature = self.conv(obs)
        q = self.q(torch.cat([feature, act], dim=-1))
        return torch.squeeze(q, -1)



class ActionValueDis(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim  = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']
        num_feature = _get_conv_out_size(obs_dim)
        self.q = mlp([num_feature] + list(hidden_sizes) + [act_dim], nn.ReLU)

    def forward(self, obs):
        feature = self.conv(obs)
        return self.q(feature)


class StateValue(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        hidden_sizes = kwargs['hidden_sizes']
        num_feature = _get_conv_out_size(obs_dim)
        self.v = mlp([num_feature] + list(hidden_sizes) + [1], get_activation_func(kwargs['hidden_activation']))

    def forward(self, obs, act):
        feature = self.conv(obs)
        v = self.v(feature)
        return torch.squeeze(v, -1)