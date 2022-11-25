#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
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

import numpy as np  # Matrix computation library
import torch
import torch.nn as nn
from gops.utils.common_utils import get_activation_func
from gops.utils.act_distribution_cls import Action_Distribution
from gops.apprfunc.ltc_cell.ltc_cell import LTCCell
from gops.apprfunc.ltc_cell.wirings import FullyConnected

# Define MLP function
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def ltc(input_dim, units, output_dim):
    wiring = FullyConnected(units, output_dim)
    ltc_cell = LTCCell(wiring, input_dim)
    return ltc_cell


def mlp_ltc(mlp_units, mlp_activation, ltc_units, input_dim, output_dim):

    if mlp_units:
        ltc_input_dim = mlp_units[-1]
        pi_sizes = [input_dim] + list(mlp_units)
        mlp_layer = mlp(
            pi_sizes,
            get_activation_func(mlp_activation))
    else:
        ltc_input_dim = input_dim
        mlp_layer = nn.Identity()
    ltc_output_dim = output_dim

    ltc_layer = ltc(ltc_input_dim, ltc_units, ltc_output_dim)
    return nn.Sequential(mlp_layer, ltc_layer)

# Count parameter number of MLP
def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


# Deterministic policy
class DetermPolicy(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        mlp_units = kwargs["mlp_units"]
        mlp_activation = kwargs["mlp_activation"]
        ltc_units = kwargs["ltc_units"]

        self.mlp_ltc = mlp_ltc(mlp_units,mlp_activation, ltc_units, obs_dim, act_dim)
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        output, next_state = self.mlp_ltc(obs)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(output) + \
                 (self.act_high_lim + self.act_low_lim) / 2
        return action


class FiniteHorizonPolicy(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"] + 1
        act_dim = kwargs["act_dim"]
        mlp_units = kwargs["mlp_units"]
        mlp_activation = kwargs["mlp_activation"]
        ltc_units = kwargs["ltc_units"]

        self.mlp_ltc = mlp_ltc(mlp_units,mlp_activation, ltc_units,obs_dim,act_dim)

        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distirbution_cls = kwargs["action_distirbution_cls"]

    def forward(self, obs, virtual_t=1):
        virtual_t = virtual_t * torch.ones(size=[obs.shape[0], 1], dtype=torch.float32, device=obs.device)
        expand_obs = torch.cat((obs, virtual_t), 1)
        output, next_state = self.mlp_ltc(expand_obs)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
            output
        ) + (self.act_high_lim + self.act_low_lim) / 2
        return action


# Stochastic Policy
class StochaPolicy(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        mlp_units = kwargs["mlp_units"]
        mlp_activation = kwargs["mlp_activation"]
        ltc_units = kwargs["ltc_units"]
        self.std_sype = kwargs["std_sype"]

        if self.std_sype == "mlp_separated":
            self.mean = mlp_ltc(mlp_units,mlp_activation, ltc_units, obs_dim, act_dim)
            self.log_std = mlp_ltc(mlp_units,mlp_activation, ltc_units, obs_dim, act_dim)
        elif self.std_sype == "mlp_shared":
            self.policy = mlp_ltc(mlp_units,mlp_activation, ltc_units, obs_dim, 2*act_dim)
        elif self.std_sype == "parameter":
            self.mean = mlp_ltc(mlp_units,mlp_activation, ltc_units, obs_dim, act_dim)
            self.log_std = nn.Parameter(torch.zeros(1, act_dim))

        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distirbution_cls = kwargs["action_distirbution_cls"]

    def forward(self, obs):
        if self.std_sype == "mlp_separated":
            action_mean, _ = self.mean(obs)
            action_std = torch.clamp(
                self.log_std(obs)[0], self.min_log_std, self.max_log_std
            ).exp()
        elif self.std_sype == "mlp_shared":
            logits,_ = self.policy(obs)
            action_mean, action_log_std = torch.chunk(logits, chunks=2, dim=-1)  # output the mean
            action_std = torch.clamp(
                action_log_std, self.min_log_std, self.max_log_std
            ).exp()
        elif self.std_sype == "parameter":
            action_mean,_ = self.mean(obs)
            action_log_std = self.log_std + torch.zeros_like(action_mean)
            action_std = torch.clamp(
                action_log_std, self.min_log_std, self.max_log_std
            ).exp()

        return torch.cat((action_mean, action_std), dim=-1)


class ActionValue(nn.Module, Action_Distribution):
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
        self.action_distirbution_cls = kwargs["action_distirbution_cls"]

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


class ActionValueDis(nn.Module, Action_Distribution):
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
        self.action_distirbution_cls = kwargs["action_distirbution_cls"]

    def forward(self, obs):
        return self.q(obs)


class ActionValueDistri(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.mean = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [1],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.denominator = max(abs(self.min_log_std), self.max_log_std)

        self.log_std = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [1],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )

    def forward(self, obs, act, min=False):
        value_mean = self.mean(torch.cat([obs, act], dim=-1))
        log_std = self.log_std(torch.cat([obs, act], dim=-1))

        value_log_std = torch.clamp_min(
            self.max_log_std * torch.tanh(log_std / self.denominator), 0
        ) + torch.clamp_max(
            -self.min_log_std * torch.tanh(log_std / self.denominator), 0
        )
        return torch.cat((value_mean, value_log_std), dim=-1)


class StochaPolicyDis(ActionValueDis, Action_Distribution):
    pass


class StateValue(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.v = mlp(
            [obs_dim] + list(hidden_sizes) + [1],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.action_distirbution_cls = kwargs["action_distirbution_cls"]

    def forward(self, obs):
        v = self.v(obs)
        return torch.squeeze(v, -1)

#
