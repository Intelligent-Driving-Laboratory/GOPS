#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Deep Deterministic Policy Gradient (DDPG) algorithm
#  Reference: Lillicrap T, Hunt J, Pritzel A et al 
#             Continuous control with deep reinforcement learning. 
#             ICLR(2016)，San Juan, Puerto Rico. 
#  Update: 2020-11-03, Hao Sun: create ddpg
#  Update: 2022-12-03, Yujie Yang: rewrite ddpg class

__all__ = ["ApproxContainer", "DDPG"]

from copy import deepcopy
from typing import Tuple

import torch
from torch.optim import Adam
import time
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.common_utils import get_apprfunc_dict
from gops.utils.tensorboard_setup import tb_tags
from gops.algorithm.base import AlgorithmBase, ApprBase


class ApproxContainer(ApprBase):
    def __init__(self, **kwargs):
        """Approximate function container for DDPG.

        Contains a policy and an action value.
        """
        super().__init__(**kwargs)
        # create value network
        q_args = get_apprfunc_dict("value", **kwargs)
        self.q = create_apprfunc(**q_args)

        # create policy network
        policy_args = get_apprfunc_dict("policy", **kwargs)
        self.policy = create_apprfunc(**policy_args)

        #  create target networks
        self.q_target = deepcopy(self.q)
        self.policy_target = deepcopy(self.policy)

        # set target network gradients
        for p in self.q_target.parameters():
            p.requires_grad = False
        for p in self.policy_target.parameters():
            p.requires_grad = False

        # set optimizers
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=kwargs["policy_learning_rate"]
        )
        self.q_optimizer = Adam(self.q.parameters(), lr=kwargs["value_learning_rate"])

    # create action_distributions
    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)


class DDPG(AlgorithmBase):
    """
    Deep Deterministic Policy Gradient (DDPG) algorithm

    Paper: https://arxiv.org/pdf/1509.02971.pdf

    Args:
        string  buffer_name : buffer type. Default to 'replay_buffer'.
        float   gamma       : discount factor. Default to 0.99.
        float   tau         : param for soft update of target network. Default to 0.005.
        int     delay_update: delay update steps for actor. Default to 1.
    """

    def __init__(self, index=0, buffer_name="replay_buffer", **kwargs):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.gamma = 0.99
        self.tau = 0.005
        self.delay_update = 1
        self.per_flag = buffer_name == "prioritized_replay_buffer"

    @property
    def adjustable_parameters(self):
        return (
            "gamma",
            "tau",
            "delay_update",
        )

    def __compute_gradient(self, data: dict, iteration):
        tb_info = dict()
        start_time = time.perf_counter()

        self.networks.q_optimizer.zero_grad()
        if not self.per_flag:
            o, a, r, o2, d = (
                data["obs"],
                data["act"],
                data["rew"],
                data["obs2"],
                data["done"],
            )
            loss_q, q = self.__compute_loss_q(o, a, r, o2, d)
            loss_q.backward()
        else:
            o, a, r, o2, d, idx, weight = (
                data["obs"],
                data["act"],
                data["rew"],
                data["obs2"],
                data["done"],
                data["idx"],
                data["weight"],
            )
            loss_q, q, abs_err = self.__compute_loss_q_per(o, a, r, o2, d, idx, weight)
            loss_q.backward()

        for p in self.networks.q.parameters():
            p.requires_grad = False

        self.networks.policy_optimizer.zero_grad()
        loss_policy = self.__compute_loss_policy(o)
        loss_policy.backward()

        for p in self.networks.q.parameters():
            p.requires_grad = True

        # record info needed in tensorboard
        end_time = time.perf_counter()
        tb_info[tb_tags["loss_critic"]] = loss_q.item()
        tb_info[tb_tags["critic_avg_value"]] = q.item()
        tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000
        tb_info[tb_tags["loss_actor"]] = loss_policy.item()

        if self.per_flag:
            return tb_info, idx, abs_err
        else:
            return tb_info

    def __compute_loss_q(self, o, a, r, o2, d):
        # Q-values
        q = self.networks.q(o, a)

        # Target Q-values
        q_policy_targ = self.networks.q_target(o2, self.networks.policy_target(o2))
        backup = r + self.gamma * (1 - d) * q_policy_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()
        return loss_q, torch.mean(q)

    def __compute_loss_q_per(self, o, a, r, o2, d, idx, weight):
        # Q-values
        q = self.networks.q(o, a)

        # Target Q-values
        q_policy_targ = self.networks.q_target(o2, self.networks.policy_target(o2))
        backup = r + self.gamma * (1 - d) * q_policy_targ

        # MSE loss against Bellman backup
        loss_q = (weight * ((q - backup) ** 2)).mean()
        abs_err = torch.abs(q - backup)
        return loss_q, torch.mean(q), abs_err

    def __compute_loss_policy(self, o):
        q_policy = self.networks.q(o, self.networks.policy(o))
        return -q_policy.mean()

    def __update(self, iteration):
        polyak = 1 - self.tau
        delay_update = self.delay_update

        self.networks.q_optimizer.step()
        if iteration % delay_update == 0:
            self.networks.policy_optimizer.step()

        with torch.no_grad():
            for p, p_targ in zip(
                self.networks.q.parameters(), self.networks.q_target.parameters()
            ):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
            for p, p_targ in zip(
                self.networks.policy.parameters(),
                self.networks.policy_target.parameters(),
            ):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def local_update(self, data: dict, iteration: int):
        extra_info = self.__compute_gradient(data, iteration)
        self.__update(iteration)
        return extra_info

    def get_remote_update_info(self, data: dict, iteration: int) -> Tuple[dict, dict]:
        extra_info = self.__compute_gradient(data, iteration)

        q_grad = [p._grad for p in self.networks.q.parameters()]
        policy_grad = [p._grad for p in self.networks.policy.parameters()]

        update_info = dict()
        update_info["q_grad"] = q_grad
        update_info["policy_grad"] = policy_grad
        update_info["iteration"] = iteration

        return extra_info, update_info

    def remote_update(self, update_info: dict):
        iteration = update_info["iteration"]
        q_grad = update_info["q_grad"]
        policy_grad = update_info["policy_grad"]

        for p, grad in zip(self.networks.q.parameters(), q_grad):
            p._grad = grad
        for p, grad in zip(self.networks.policy.parameters(), policy_grad):
            p._grad = grad

        self.__update(iteration)
