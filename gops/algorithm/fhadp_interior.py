#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

__all__ = ["FHADPInterior"]

from typing import Tuple

import torch
from gops.algorithm.fhadp import ApproxContainer, FHADP
from gops.utils.gops_typing import DataDict, InfoDict
from gops.utils.tensorboard_setup import tb_tags

EPSILON = 1e-8


class FHADPInterior(FHADP):
    def __init__(
        self,
        *,
        pre_horizon: int,
        gamma: float = 1.0,
        penalty: float = 1.0,
        penalty_increase: float = 1.1,
        penalty_delay: float = 100,
        max_penalty: float = 1e3,
        index: int = 0,
        **kwargs,
    ):
        super().__init__(
            pre_horizon=pre_horizon,
            gamma=gamma,
            index=index,
            **kwargs,
        )
        self.penalty = penalty
        self.penalty_increase = penalty_increase
        self.penalty_delay = penalty_delay
        self.max_penalty = max_penalty
        self.update_step = 0

    @property
    def adjustable_parameters(self) -> Tuple[str]:
        return (
            *super().adjustable_parameters,
            "penalty",
            "penalty_increase",
            "penalty_delay",
        )

    def _compute_loss_policy(self, data: DataDict) -> Tuple[torch.Tensor, InfoDict]:
        o, d = data["obs"], data["done"]
        info = data
        v_pi_r = 0
        constraint = []
        v_pi_c_int = 0
        v_pi_c_ext = 0
        for step in range(self.pre_horizon):
            a = self.networks.policy(o, step + 1)
            o, r, d, info = self.envmodel.forward(o, a, d, info)
            c = info["constraint"]
            constraint.append(c)
            c_int = (-torch.clamp_max(c, 0) + EPSILON).log().sum(1)
            c_ext = (torch.clamp_min(c, 0) ** 2).sum(1)
            v_pi_r += r * (self.gamma ** step)
            v_pi_c_int += c_int * (self.gamma ** step)
            v_pi_c_ext += c_ext * (self.gamma ** step)
        constraint = torch.stack(constraint, dim=1)
        feasible = (constraint < 0).all(2).all(1)
        loss_reward = -v_pi_r.mean()
        loss_constraint_int = (v_pi_c_int * feasible).mean()
        loss_constraint_ext = (v_pi_c_ext * ~feasible).mean()
        loss_policy = loss_reward + \
            1 / self.penalty * loss_constraint_int + \
            self.penalty * loss_constraint_ext

        self.update_step += 1
        if self.update_step % self.penalty_delay == 0:
            self.penalty = min(self.penalty * self.penalty_increase, self.max_penalty)

        loss_info = {
            tb_tags["loss_actor"]: loss_policy.item(),
            tb_tags["loss_actor_reward"]: loss_reward.item(),
            tb_tags["loss_actor_constraint"]: loss_constraint_ext.item(),
            "Loss/Penalty coefficient-RL iter": self.penalty,
            "Loss/Feasible ratio-RL iter": feasible.float().mean().item(),
        }
        return loss_policy, loss_info
