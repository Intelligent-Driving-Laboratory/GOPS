#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description:  NCP
#  Update: 2022-11-25, Yinuo Wang : create NCP function

__all__ = [
    "DetermPolicy",
    # "FiniteHorizonPolicy",
    # "StochaPolicy",
    # "ActionValue",
    # "ActionValueDis",
    # "ActionValueDistri",
    # "StateValue",
]



import numpy as np  # Matrix computation library
import torch
import torch.nn as nn
from gops.utils.common_utils import get_activation_func
from gops.utils.act_distribution_cls import Action_Distribution
from ncps.wirings import AutoNCP
from ncps.torch import LTC

# Define MLP function
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

# Define NCP function
def ncp(ncp_units, act_dim, obs_dim):
    wiring = AutoNCP(ncp_units, act_dim)  #  units, 1 motor neuron
    ltc_model = LTC(obs_dim, wiring, batch_first=True)
    return ltc_model

# Build MLP-NCP Network
def mlp_ncp(mlp_units, mlp_activation, ltc_units, input_dim, output_dim):
    if mlp_units:
        ncp_input_dim = mlp_units[-1]
        pi_sizes = [input_dim] + list(mlp_units)
        mlp_layer = mlp(
            pi_sizes,
            get_activation_func(mlp_activation))
    else:
        ncp_input_dim = input_dim
        mlp_layer = nn.Identity()
    ncp_output_dim = output_dim

    ncp_layer = ncp(ltc_units, ncp_output_dim, ncp_input_dim)
    return nn.Sequential(mlp_layer, ncp_layer)

class DetermPolicy(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        mlp_units = kwargs["mlp_units"]
        mlp_activation = kwargs["mlp_activation"]
        ncp_units = kwargs["ncp_units"]

        self.pi = mlp_ncp(mlp_units, mlp_activation, ncp_units, obs_dim, act_dim)
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        obs = torch.unsqueeze(obs, dim=1)
        model_out, _ = self.pi.forward(obs)
        model_out = torch.squeeze(model_out, dim=1)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
            model_out
        ) + (self.act_high_lim + self.act_low_lim) / 2
        return action