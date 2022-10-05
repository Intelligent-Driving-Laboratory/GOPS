#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yao MU
#  Description: Structural definition for approximation function
#
#  Update Date: 2021-05-21, Shengbo Li: revise headline

__all__ = [
    "DetermPolicy",
    "StochaPolicy",
    "ActionValue",
    "ActionValueDis",
    "StateValue",
]


import numpy as np
import torch
import torch.nn as nn
from math import factorial
from gops.utils.act_distribution_cls import Action_Distribution


def make_features(x, degree):  # TODO: More concise
    def n_matmul(x, n):
        def matmul_crossing(a, b):
            batchsize = a.size(0)
            return torch.matmul(
                torch.transpose(a.unsqueeze(1), -1, -2), b.unsqueeze(1)
            ).reshape(batchsize, -1)

        a = x
        b = x
        if n == 0:
            return torch.ones_like(a)
        for _ in range(n - 1):
            a = matmul_crossing(a, b)
        return a

    return torch.cat([n_matmul(x, i) for i in range(0, degree)], 1)


def get_features_dim(input_dim, degree):
    x = torch.zeros([1, input_dim])
    return make_features(x, degree).size(1)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def combination(m, n):
    return int(factorial(m) / (factorial(n) * factorial(m - n)))


def create_features(x, degree=2):
    batch = x.shape[0]
    obs_dim = x.shape[1]
    if degree == 2:
        features_dim = combination(degree + obs_dim - 1, degree)
    else:
        raise ValueError("Not set degree properly")
    features = torch.zeros((batch, features_dim))

    if degree == 2:
        k = 0
        for i in range(0, obs_dim):
            for j in range(i, obs_dim):
                features[:, k:k + 1] = torch.mul(x[:, i:i + 1], x[:, j:j + 1])
                k = k + 1
    else:
        raise ValueError("Not set degree properly")

    return features


def count_features_dim(input_dim, degree):
    x = torch.zeros([1, input_dim])
    return create_features(x, degree).size(1)


class DetermPolicy(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        self.degree = kwargs["degree"]
        self.pi = nn.Linear(get_features_dim(obs_dim, self.degree), act_dim)
        action_high_limit = kwargs["act_high_lim"]
        action_low_limit = kwargs["act_low_lim"]
        self.register_buffer("act_high_lim", torch.from_numpy(action_high_limit))
        self.register_buffer("act_low_lim", torch.from_numpy(action_low_limit))
        self.action_distirbution_cls = kwargs["action_distirbution_cls"]

    def forward(self, obs):
        obs = make_features(obs, self.degree)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
            self.pi(obs)
        ) + (self.act_high_lim + self.act_low_lim) / 2
        return action


class StochaPolicy(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        action_high_limit = kwargs["act_high_lim"]
        action_low_limit = kwargs["act_low_lim"]
        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.degree = kwargs["degree"]
        self.mean = nn.Linear(get_features_dim(obs_dim, self.degree), act_dim)
        self.log_std = nn.Linear(get_features_dim(obs_dim, self.degree), act_dim)
        self.register_buffer("act_high_lim", torch.from_numpy(action_high_limit))
        self.register_buffer("act_low_lim", torch.from_numpy(action_low_limit))
        self.action_distirbution_cls = kwargs["action_distirbution_cls"]

    def forward(self, obs):
        obs = make_features(obs, self.degree)
        action_mean = self.mean(obs)
        action_std = torch.clamp(
            self.log_std(obs), self.min_log_std, self.max_log_std
        ).exp()
        return torch.cat((action_mean, action_std), dim=-1)


class ActionValue(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        self.degree = kwargs["degree"]
        self.q = nn.Linear(get_features_dim(obs_dim + act_dim, self.degree), act_dim)
        self.action_distirbution_cls = kwargs["action_distirbution_cls"]

    def forward(self, obs, act):
        input = torch.cat([obs, act], dim=-1)
        input = make_features(input, self.degree)
        q = self.q(input)
        return torch.squeeze(q, -1)


class ActionValueDis(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_num = kwargs["act_num"]
        self.degree = kwargs["degree"]
        self.q = nn.Linear(get_features_dim(obs_dim, self.degree), act_num)
        self.action_distirbution_cls = kwargs["action_distirbution_cls"]

    def forward(self, obs):
        obs = make_features(obs, self.degree)
        return self.q(obs)


class StochaPolicyDis(ActionValueDis, Action_Distribution):
    pass


class StateValue(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        self.degree = kwargs["degree"]
        self.v = nn.Linear(count_features_dim(obs_dim, self.degree), 1)
        self.action_distirbution_cls = kwargs["action_distirbution_cls"]

    def forward(self, obs):
        obs = create_features(obs, self.degree)
        return self.v(obs)


if __name__ == "__main__":
    obs = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    print(make_features(obs, 1))


if __name__ == "__main__":
    obs = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    print(make_features(obs, 1))
