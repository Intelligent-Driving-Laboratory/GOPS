#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

__all__ = ["FHADPLagrangiannet"]


from typing import Tuple

import torch
from torch.optim import Adam
from gops.algorithm.fhadp import FHADP
from gops.utils.gops_typing import DataDict, InfoDict
from gops.utils.tensorboard_setup import tb_tags
from gops.utils.common_utils import get_apprfunc_dict
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.algorithm.base import ApprBase


class ApproxContainer(ApprBase):
    def __init__(
        self,
        *,
        policy_learning_rate: float,
        **kwargs,
    ):
        """Approximate function container for FHADP."""
        """Contains one policy network."""
        super().__init__(**kwargs)
        policy_args = get_apprfunc_dict("policy", **kwargs)
        multiplier_args = get_apprfunc_dict("multiplier", **kwargs)
        self.policy = create_apprfunc(**policy_args)
        self.multiplier_net = create_apprfunc(**multiplier_args)
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=policy_learning_rate
        )
        self.mutiplier_optimizer = Adam(
            self.multiplier_net.parameters(), lr=policy_learning_rate*0.1
        )
        self.optimizer_dict = {
            "policy": self.policy_optimizer,
        }
        self.init_scheduler(**kwargs)

    def create_action_distributions(self, logits):
        """create action distribution"""
        return self.policy.get_act_dist(logits)



class FHADPLagrangiannet(FHADP):
    def __init__(
        self,
        *,
        pre_horizon: int,
        gamma: float = 1.0,
        multiplier_delay: int = 1,
        index: int = 0,
        **kwargs,
    ):
        super().__init__(
            pre_horizon=pre_horizon,
            gamma=gamma,
            index=index,
            **kwargs,
        )

        self.networks = ApproxContainer(**kwargs)
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
        multiplier_list = []
        l_list = []
        c_list = []
        for step in range(self.pre_horizon):
            a = self.networks.policy(o, step + 1)
            multiplier = self.networks.multiplier_net(o.detach(), step + 1)
            multiplier = torch.nn.functional.softplus(
                100.0 * torch.tanh(multiplier)
            )
            o, r, d, info = self.envmodel.forward(o, a, d, info)
            c = torch.clamp_min(info["constraint"], 0.)
            l_list.append(-r * (self.gamma ** step))
            c_list.append(c * (self.gamma ** step))
            multiplier_list.append(multiplier)
        
        # cal loss policy
        loss_policy = 0.
        loss_multiplier = 0.
        for step in range(self.pre_horizon):
            loss_policy += (l_list[step] + multiplier_list[step].detach() * c_list[step]).mean()
            loss_multiplier -= (multiplier_list[step] * c_list[step].detach()).mean()

        self.update_step += 1
        if  self.update_step % self.multiplier_delay == 0:
            self.networks.mutiplier_optimizer.zero_grad()
            loss_multiplier.backward()
            self.networks.mutiplier_optimizer.step()

        loss_reward = torch.sum(torch.stack(l_list, dim=1), dim=1).mean()
        c_clip_list = []
        for step in range(self.pre_horizon):
            c_clip_list.append(torch.clamp(c_list[step], min=0.))
        loss_constraint = torch.sum(torch.stack(c_clip_list, dim=1), dim=1).mean()
        avg_multiplier = torch.mean(torch.stack(multiplier_list, dim=1), dim=1).mean()

        loss_info = {
            tb_tags["loss_actor"]: loss_policy.item(),
            tb_tags["loss_actor_reward"]: loss_reward.item(),
            tb_tags["loss_actor_constraint"]: loss_constraint.item(),
            "Loss/Lagrange multiplier-RL iter": avg_multiplier.item(),
        }
        return loss_policy, loss_info