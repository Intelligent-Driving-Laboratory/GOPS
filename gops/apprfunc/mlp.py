#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Multilayer Perceptron (MLP)
#  Update: 2021-03-05, Wenjun Zou: create MLP function


__all__ = [
    "DetermPolicy",
    "FiniteHorizonPolicy",
    "StochaPolicy",
    "ActionValue",
    "ActionValueDis",
    "ActionValueDistri",
    "StateValue",
]

import numpy as np
import torch
import torch.nn as nn
from gops.utils.common_utils import get_activation_func
from gops.utils.act_distribution_cls import Action_Distribution


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
        hidden_sizes = kwargs["hidden_sizes"]

        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
            self.pi(obs)
        ) + (self.act_high_lim + self.act_low_lim) / 2
        return action


class FiniteHorizonPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy for finite-horizon.
    Input: observation, time step.
    Output: action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"] + 1
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]

        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs, virtual_t=1):
        virtual_t = virtual_t * torch.ones(
            size=[obs.shape[0], 1], dtype=torch.float32, device=obs.device
        )
        expand_obs = torch.cat((obs, virtual_t), 1)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
            self.pi(expand_obs)
        ) + (self.act_high_lim + self.act_low_lim) / 2
        return action


# Stochastic Policy
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
        hidden_sizes = kwargs["hidden_sizes"]
        self.std_type = kwargs["std_type"]

        # mean and log_std are calculated by different MLP
        if self.std_type == "mlp_separated":
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
            self.mean = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
            self.log_std = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
        # mean and log_std are calculated by same MLP
        elif self.std_type == "mlp_shared":
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim * 2]
            self.policy = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
        # mean is calculated by MLP, and log_std is learnable parameter
        elif self.std_type == "parameter":
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
            self.mean = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
            self.log_std = nn.Parameter(-0.5*torch.ones(1, act_dim))

        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        if self.std_type == "mlp_separated":
            action_mean = self.mean(obs)
            action_std = torch.clamp(
                self.log_std(obs), self.min_log_std, self.max_log_std
            ).exp()
        elif self.std_type == "mlp_shared":
            logits = self.policy(obs)
            action_mean, action_log_std = torch.chunk(
                logits, chunks=2, dim=-1
            )  # output the mean
            action_std = torch.clamp(
                action_log_std, self.min_log_std, self.max_log_std
            ).exp()
        elif self.std_type == "parameter":
            action_mean = self.mean(obs)
            action_log_std = self.log_std + torch.zeros_like(action_mean)
            action_std = torch.clamp(
                action_log_std, self.min_log_std, self.max_log_std
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
        hidden_sizes = kwargs["hidden_sizes"]
        self.q = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [1],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
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
        hidden_sizes = kwargs["hidden_sizes"]
        self.q = mlp(
            [obs_dim] + list(hidden_sizes) + [act_num],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        return self.q(obs)


class ActionValueDistri(nn.Module):
    """
    Approximated function of distributed action-value function.
    Input: observation.
    Output: parameters of action-value distribution.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.q = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [2],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.denominator = max(abs(self.min_log_std), self.max_log_std)

    def forward(self, obs, act, min=False):
        logits = self.q(torch.cat([obs, act], dim=-1))
        value_mean, log_std = torch.chunk(logits, chunks=2, dim=-1)

        value_log_std = torch.clamp_min(
            self.max_log_std * torch.tanh(log_std / self.denominator), 0
        ) + torch.clamp_max(
            -self.min_log_std * torch.tanh(log_std / self.denominator), 0
        )
        return torch.cat((value_mean, value_log_std), dim=-1)


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
        hidden_sizes = kwargs["hidden_sizes"]
        self.v = mlp(
            [obs_dim] + list(hidden_sizes) + [1],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        v = self.v(obs)
        return torch.squeeze(v, -1)
