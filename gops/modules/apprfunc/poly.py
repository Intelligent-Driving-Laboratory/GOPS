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

def make_features(x,degree=4):
    batch = x.shape[0]
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(0, degree)], 1).reshape(batch,-1)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class DetermPolicy(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        hidden_sizes = kwargs['hidden_sizes']
        act_limit = kwargs['action_high_limit']
        self.degree=4
        self.pi = nn.Linear(obs_dim*self.degree,act_dim)
        self.act_limit =   torch.from_numpy(act_limit)

    def forward(self, obs):
        obs=make_features(obs,self.degree)
        return self.act_limit * torch.tanh(self.pi(obs))

class StochaPolicy(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']*2
        hidden_sizes = kwargs['hidden_sizes']
        act_limit = kwargs['action_high_limit']
        self.degree = 4
        self.mean = nn.Linear(obs_dim*self.degree,act_dim)
        self.std = nn.Linear(obs_dim * self.degree, act_dim)
        self.act_limit =   torch.from_numpy(act_limit)

    def forward(self, obs):
        obs=make_features(obs,self.degree)
        return self.act_limit * torch.tanh(self.mean(obs)), torch.exp(self.std(obs))


class ActionValue(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        self.degree = 2
        self.q = nn.Linear((obs_dim+act_dim) * self.degree, 1)

    def forward(self, obs, act):
        input=torch.cat([obs, act], dim=-1)
        input = make_features(input, self.degree)
        q = self.q(input)
        return torch.squeeze(q, -1)

class ActionValueDis(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim  = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        self.degree = 2
        self.q = nn.Linear(obs_dim*self.degree,act_dim)

    def forward(self, obs):
        obs = make_features(obs, self.degree)
        return self.q(obs)





class StateValue(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs['obs_dim']
        self.v = nn.Linear((obs_dim) * self.degree, 1)

    def forward(self, obs):
        input = make_features(obs, self.degree)
        return self.v(obs)
