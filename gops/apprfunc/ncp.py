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


# Define NCP function
def ncp(ncp_units, act_dim, obs_dim):
    wiring = AutoNCP(ncp_units, act_dim)  #  units, 1 motor neuron
    ltc_model = LTC(obs_dim, wiring, batch_first=True)
    return ltc_model


class DetermPolicy(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super().__init__()
        ncp_units = kwargs["ncp_units"]
        act_dim = kwargs["act_dim"]
        obs_dim = kwargs["obs_dim"]

        self.pi = ncp(ncp_units, act_dim, obs_dim)
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