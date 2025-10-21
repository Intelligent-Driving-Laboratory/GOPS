#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Feasible Policy Iteration with Soft Actor-Critic (FPISAC) algorithm
#  Reference: Yang Y, Zheng Z, Li SE et al (2025) 
#             Feasible Policy Iteration for Safe Reinforcement Learning.
#             arXiv preprint arXiv:2304.08845.
#  Update: 2025-10-21, Yujie Yang: create FPISAC algorithm

__all__ = ["ApproxContainer", "FPISAC"]

import math
import time
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from gops.algorithm.base import AlgorithmBase
from gops.algorithm.sac import ApproxContainer as SACApproxContainer, SAC
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.common_utils import get_apprfunc_dict
from gops.utils.gops_typing import DataDict
from gops.utils.math_utils import incremental_update
from gops.utils.tensorboard_setup import tb_tags


class ApproxContainer(SACApproxContainer):
    def __init__(
        self,
        value_learning_rate: float,
        feasibility_learning_rate: float,
        policy_learning_rate: float,
        alpha_learning_rate: float,
        **kwargs,
    ):
        super().__init__(
            q_learning_rate=value_learning_rate,
            policy_learning_rate=policy_learning_rate,
            alpha_learning_rate=alpha_learning_rate,
            **kwargs,
        )
        # create feasibility networks
        g_args = get_apprfunc_dict("feasibility", **kwargs)
        self.g1: nn.Module = create_apprfunc(**g_args)
        self.g2: nn.Module = create_apprfunc(**g_args)
        self.g1_target: nn.Module = deepcopy(self.g1)
        self.g2_target: nn.Module = deepcopy(self.g2)

        for p in self.g1_target.parameters():
            p.requires_grad = False
        for p in self.g2_target.parameters():
            p.requires_grad = False

        self.g1_optimizer = Adam(self.g1.parameters(), lr=feasibility_learning_rate)
        self.g2_optimizer = Adam(self.g2.parameters(), lr=feasibility_learning_rate)


class FPISAC(SAC):
    def __init__(
        self,
        index: int = 0,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 1.,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        gamma_g: float = 0.99,
        epsilon: float = 0.1,
        penalty: float = 1.,
        **kwargs,
    ):
        AlgorithmBase.__init__(self, index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.networks.log_alpha.data.fill_(math.log(alpha))
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        if target_entropy is None:
            self.target_entropy = -kwargs["action_dim"]
        else:
            self.target_entropy = target_entropy
        self.gamma_g = gamma_g
        self.epsilon = epsilon
        self.penalty = penalty

    @property
    def adjustable_parameters(self):
        return super().adjustable_parameters + ("gamma_g", "epsilon", "penalty")

    def _compute_gradient(self, data: DataDict, iteration: int):
        start_time = time.time()

        obs = data["obs"]
        logits = self.networks.policy(obs)
        act_dist = self.networks.create_action_distributions(logits)
        new_act, new_logp = act_dist.rsample()
        data.update({"new_act": new_act, "new_logp": new_logp})

        self.networks.q1_optimizer.zero_grad()
        self.networks.q2_optimizer.zero_grad()
        loss_q, q1, q2 = self._compute_loss_q(data)
        loss_q.backward()

        self.networks.g1_optimizer.zero_grad()
        self.networks.g2_optimizer.zero_grad()
        loss_g, g1, g2 = self._compute_loss_g(data)
        loss_g.backward()

        for p in self.networks.q1.parameters():
            p.requires_grad = False
        for p in self.networks.q2.parameters():
            p.requires_grad = False
        for p in self.networks.g1.parameters():
            p.requires_grad = False
        for p in self.networks.g2.parameters():
            p.requires_grad = False

        self.networks.policy_optimizer.zero_grad()
        loss_policy, (entropy, fea) = self._compute_loss_policy(data)
        loss_policy.backward()

        for p in self.networks.q1.parameters():
            p.requires_grad = True
        for p in self.networks.q2.parameters():
            p.requires_grad = True
        for p in self.networks.g1.parameters():
            p.requires_grad = True
        for p in self.networks.g2.parameters():
            p.requires_grad = True

        if self.auto_alpha:
            self.networks.alpha_optimizer.zero_grad()
            loss_alpha = -self.networks.log_alpha * (self.target_entropy - entropy)
            loss_alpha.backward()

        tb_info = {
            tb_tags["loss_critic"]: loss_q.item(),
            tb_tags["loss_actor"]: loss_policy.item(),
            "Loss/Scenery loss-RL iter": loss_g.item(),
            "FPISAC/critic_avg_q1-RL iter": q1.item(),
            "FPISAC/critic_avg_q2-RL iter": q2.item(),
            "FPISAC/scenery_avg_g1-RL iter": g1.item(),
            "FPISAC/scenery_avg_g2-RL iter": g2.item(),
            "FPISAC/entropy-RL iter": entropy.item(),
            "FPISAC/alpha-RL iter": self._get_alpha(),
            "FPISAC/violation-RL iter": data["next_constraint"].mean().item(),
            "FPISAC/feasible-RL iter": fea.item(),
            tb_tags["alg_time"]: (time.time() - start_time) * 1000,
        }

        return tb_info

    def _compute_loss_g(self, data: DataDict):
        obs, act, constraint, obs2, done = (
            data["obs"],
            data["act"],
            data["next_constraint"],
            data["obs2"],
            data["done"],
        )
        g1 = self.networks.g1(obs, act)
        g2 = self.networks.g2(obs, act)
        with torch.no_grad():
            next_logits = self.networks.policy(obs2)
            next_act_dist = self.networks.create_action_distributions(next_logits)
            next_act, _ = next_act_dist.sample()
            g1_next = self.networks.g1_target(obs2, next_act)
            g2_next = self.networks.g2_target(obs2, next_act)
            g_next = torch.clamp(torch.max(g1_next, g2_next), 0, 1)
            target_g = constraint + (1 - done) * (1 - constraint) * self.gamma_g * g_next
        g1_loss = ((g1 - target_g) ** 2).mean()
        g2_loss = ((g2 - target_g) ** 2).mean()
        return g1_loss + g2_loss, g1.mean().detach(), g2.mean().detach()

    def _compute_loss_policy(self, data: DataDict):
        obs, new_act, new_logp = (
            data["obs"],
            data["new_act"],
            data["new_logp"],
        )

        g1 = self.networks.g1(obs, new_act)
        g2 = self.networks.g2(obs, new_act)
        g = torch.max(g1, g2)

        q1 = self.networks.q1(obs, new_act)
        q2 = self.networks.q2(obs, new_act)
        q = torch.min(q1, q2)

        fea = g <= self.epsilon
        loss = (self._get_alpha() * new_logp - q + self.penalty * ~fea * g).mean()
        return loss, (-new_logp.mean().detach(), fea.float().mean())

    def _update(self, iteration: int):
        super()._update(iteration)

        self.networks.g1_optimizer.step()
        self.networks.g2_optimizer.step()

        incremental_update(self.networks.g1, self.networks.g1_target, self.tau)
        incremental_update(self.networks.g2, self.networks.g2_target, self.tau)
