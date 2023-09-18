#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

__all__ = ["FHADPLagrangian"]

import math
from typing import Tuple

import torch
from torch import nn
from torch.optim import Adam
from gops.algorithm.fhadp import ApproxContainer, FHADP
from gops.utils.gops_typing import DataDict, InfoDict
from gops.utils.tensorboard_setup import tb_tags


class FHADPLagrangian(FHADP):
    def __init__(
        self,
        *,
        pre_horizon: int,
        gamma: float = 1.0,
        multiplier: float = 1.0,
        multiplier_lr: float = 1e-3,
        multiplier_delay: int = 10,
        index: int = 0,
        **kwargs,
    ):
        super().__init__(
            pre_horizon=pre_horizon,
            gamma=gamma,
            index=index,
            **kwargs,
        )
        # inverse of softplus function
        self.multiplier_param = nn.Parameter(torch.tensor(
            math.log(math.exp(multiplier) - 1), dtype=torch.float32))
        self.multiplier_optim = Adam([self.multiplier_param], lr=multiplier_lr)
        self.multiplier_delay = multiplier_delay
        self.update_step = 0

    @property
    def adjustable_parameters(self) -> Tuple[str]:
        return (
            *super().adjustable_parameters,
            "multiplier",
            "multiplier_lr",
            "multiplier_delay",
        )

    def _compute_loss_policy(self, data: DataDict) -> Tuple[torch.Tensor, InfoDict]:
        o, d = data["obs"], data["done"]
        info = data
        v_pi_r = 0
        v_pi_c = 0
        for step in range(self.pre_horizon):
            a = self.networks.policy(o, step + 1)
            o, r, d, info = self.envmodel.forward(o, a, d, info)
            c = torch.clamp_min(info["constraint"], 0).sum(1)
            v_pi_r += r * (self.gamma ** step)
            v_pi_c += c * (self.gamma ** step)
        loss_reward = -v_pi_r.mean()
        loss_constraint = v_pi_c.mean()
        multiplier = torch.nn.functional.softplus(self.multiplier_param).item()
        loss_policy = loss_reward + multiplier * loss_constraint

        self.update_step += 1
        if self.update_step % self.multiplier_delay == 0:
            multiplier_loss = -self.multiplier_param * loss_constraint.item()
            self.multiplier_optim.zero_grad()
            multiplier_loss.backward()
            self.multiplier_optim.step()

        loss_info = {
            tb_tags["loss_actor"]: loss_policy.item(),
            tb_tags["loss_actor_reward"]: loss_reward.item(),
            tb_tags["loss_actor_constraint"]: loss_constraint.item(),
            "Loss/Lagrange multiplier-RL iter": multiplier,
        }
        return loss_policy, loss_info
