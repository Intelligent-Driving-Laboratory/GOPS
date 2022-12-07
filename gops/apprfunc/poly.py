#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Structural definition for approximation function
#  Update: 2021-03-05, Wenjun Zou: create poly function

__all__ = [
    "DetermPolicy",
    "FiniteHorizonPolicy",
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


# Define polynomial function
# TODO: More concise
def make_features(x, degree):
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

    return torch.cat([n_matmul(x, i) for i in range(1, degree + 1)], 1)


# input_dim: dimention of state, degree: degree of polynomial function
# return dimention of feature
def get_features_dim(input_dim, degree):
    x = torch.zeros([1, input_dim])
    return make_features(x, degree).size(1)


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
                features[:, k : k + 1] = torch.mul(x[:, i : i + 1], x[:, j : j + 1])
                k = k + 1
    else:
        raise ValueError("Not set degree properly")

    return features


def count_features_dim(input_dim, degree):
    x = torch.zeros([1, input_dim])
    return create_features(x, degree).size(1)


class DetermPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy.
    Input: observation.
    Output: action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        self.degree = kwargs["degree"]
        self.add_bias = kwargs["add_bias"]
        self.pi = nn.Linear(
            get_features_dim(obs_dim, self.degree), act_dim, bias=self.add_bias
        )
        action_high_limit = kwargs["act_high_lim"]
        action_low_limit = kwargs["act_low_lim"]
        self.register_buffer("act_high_lim", torch.from_numpy(action_high_limit))
        self.register_buffer("act_low_lim", torch.from_numpy(action_low_limit))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        obs = make_features(obs, self.degree)
        # action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
        #     self.pi(obs)
        # ) + (self.act_high_lim + self.act_low_lim) / 2
        action = self.pi(obs)
        return action


class FiniteHorizonPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy for finite-horizon.
    Input: observation, time step.
    Output: action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        self.degree = kwargs["degree"]
        self.add_bias = kwargs["add_bias"]
        self.pi = nn.Linear(
            get_features_dim(obs_dim, self.degree) + 1, act_dim, bias=self.add_bias
        )
        action_high_limit = kwargs["act_high_lim"]
        action_low_limit = kwargs["act_low_lim"]
        self.register_buffer("act_high_lim", torch.from_numpy(action_high_limit))
        self.register_buffer("act_low_lim", torch.from_numpy(action_low_limit))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs, virtual_t=1):
        obs = make_features(obs, self.degree)
        virtual_t = virtual_t * torch.ones(
            size=[obs.shape[0], 1], dtype=torch.float32, device=obs.device
        )
        expand_obs = torch.cat((obs, virtual_t), 1)
        # obs = make_features(obs, self.degree)
        # action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
        #     self.pi(expand_obs)
        # ) + (self.act_high_lim + self.act_low_lim) / 2
        action = self.pi(expand_obs)
        return action


class StochaPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of stochastic policy.
    Input: observation.
    Output: parameters of action distribution.
    """

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
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        obs = make_features(obs, self.degree)
        action_mean = self.mean(obs)
        action_std = torch.clamp(
            self.log_std(obs), self.min_log_std, self.max_log_std
        ).exp()
        return torch.cat((action_mean, action_std), dim=-1)


class ActionValue(nn.Module, Action_Distribution):
    """
    Approximated function of action-value function.
    Input: observation, action.
    Output: action-value.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        self.degree = kwargs["degree"]
        self.q = nn.Linear(get_features_dim(obs_dim + act_dim, self.degree), act_dim)
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs, act):
        input = torch.cat([obs, act], dim=-1)
        input = make_features(input, self.degree)
        q = self.q(input)
        return torch.squeeze(q, -1)


class ActionValueDis(nn.Module, Action_Distribution):
    """
    Approximated function of action-value function for discrete action space.
    Input: observation.
    Output: action-value for all action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_num = kwargs["act_num"]
        self.degree = kwargs["degree"]
        self.q = nn.Linear(get_features_dim(obs_dim, self.degree), act_num)
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        obs = make_features(obs, self.degree)
        return self.q(obs)


class StochaPolicyDis(ActionValueDis, Action_Distribution):
    """
    Approximated function of stochastic policy for discrete action space.
    Input: observation.
    Output: parameters of action distribution.
    """

    pass


class StateValue(nn.Module, Action_Distribution):
    """
    Approximated function of state-value function.
    Input: observation, action.
    Output: state-value.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        self.add_bias = kwargs["add_bias"]
        if kwargs["norm_matrix"] is None:
            kwargs["norm_matrix"] = [1.0] * obs_dim
        self.norm_matrix = torch.from_numpy(
            np.array(kwargs["norm_matrix"], dtype=np.float32)
        )
        self.degree = kwargs["degree"]
        self.v = nn.Linear(
            count_features_dim(obs_dim, self.degree), 1, bias=self.add_bias
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        obs = create_features(torch.mul(obs, self.norm_matrix), self.degree)
        return self.v(obs).squeeze(-1)


if __name__ == "__main__":
    obs = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    print(make_features(obs, 1))


if __name__ == "__main__":
    obs = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    print(make_features(obs, 1))
