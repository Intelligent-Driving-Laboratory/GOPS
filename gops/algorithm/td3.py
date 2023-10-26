#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Twin Delayed Deep Deterministic policy gradient (TD3) algorithm
#  Reference: Fujimoto S, Hoof H, Meger D (2018) 
#             Addressing function approximation error in actor-critic methods. 
#             ICML, Stockholm, Sweden.
#  Update: 2021-03-05, Wenxuan Wang: create TD3 algorithm

__all__ = ["ApproxContainer", "TD3"]

import time
from copy import deepcopy
from typing import Tuple

import torch
from torch.optim import Adam

from gops.algorithm.base import AlgorithmBase, ApprBase
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.tensorboard_setup import tb_tags
from gops.utils.common_utils import get_apprfunc_dict


class ApproxContainer(ApprBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # create value network
        q_args = get_apprfunc_dict("value", **kwargs)
        self.q1 = create_apprfunc(**q_args)
        self.q2 = create_apprfunc(**q_args)

        # create policy network
        policy_args = get_apprfunc_dict("policy", **kwargs)
        self.policy = create_apprfunc(**policy_args)

        #  create target networks
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        self.policy_target = deepcopy(self.policy)

        # set target network gradients
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False
        for p in self.policy_target.parameters():
            p.requires_grad = False

        # set optimizers
        self.q1_optimizer = Adam(self.q1.parameters(), lr=kwargs["value_learning_rate"])
        self.q2_optimizer = Adam(self.q2.parameters(), lr=kwargs["value_learning_rate"])
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=kwargs["policy_learning_rate"]
        )

    # create action_distributions
    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)


class TD3(AlgorithmBase):
    """
    Twin Delayed Deep Deterministic policy gradient (TD3) algorithm

    Paper: https://arxiv.org/pdf/1802.09477.pdf

    Args:
        list    action_high_limit   : action limit for available actions.
        float   target_noise        : action noise for target pi network. Default to 0.2
        float   noise_clip          : range [-noise_clip, noise_clip] for target_noise. Default to 0.5
        string  buffer_name         : buffer type. Default to 'replay_buffer'.
        int     index               : for calculating offset of random seed for subprocess. Default to 0.
    """

    def __init__(
        self,
        target_noise=0.2,
        noise_clip=0.5,
        buffer_name="replay_buffer",
        index=0,
        **kwargs
    ):
        super(TD3, self).__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.act_low_limit = kwargs["action_low_limit"]
        self.act_high_limit = kwargs["action_high_limit"]
        self.gamma = 0.99
        self.tau = 0.005
        self.delay_update = 2
        self.reward_scale = 1
        self.per_flag = buffer_name == "prioritized_replay_buffer"

    @property
    def adjustable_parameters(self):
        para_tuple = ("gamma", "tau", "delay_update", "reward_scale")
        return para_tuple

    def __compute_gradient(self, data: dict, iteration):
        tb_info = dict()
        start_time = time.time()
        self.networks.q1_optimizer.zero_grad()
        self.networks.q2_optimizer.zero_grad()
        self.networks.policy_optimizer.zero_grad()

        if not self.per_flag:
            o, a, r, o2, d = (
                data["obs"],
                data["act"],
                data["rew"] * self.reward_scale,
                data["obs2"],
                data["done"],
            )
            loss_q, loss_q1, loss_q2 = self.__compute_loss_q(o, a, r, o2, d)
            loss_q.backward()
        else:
            o, a, r, o2, d, idx, weight = (
                data["obs"],
                data["act"],
                data["rew"] * self.reward_scale,
                data["obs2"],
                data["done"],
                data["idx"],
                data["weight"],
            )
            loss_q, loss_q1, loss_q2, abs_err = self.__compute_loss_q_per(
                o, a, r, o2, d, idx, weight
            )
            loss_q.backward()

        for p in self.networks.q1.parameters():
            p.requires_grad = False
        for p in self.networks.q2.parameters():
            p.requires_grad = False

        loss_policy = self.__compute_loss_pi(o)
        loss_policy.backward()

        for p in self.networks.q1.parameters():
            p.requires_grad = True
        for p in self.networks.q2.parameters():
            p.requires_grad = True

        end_time = time.time()
        tb_info[tb_tags["loss_critic"]] = loss_q.item()
        tb_info[tb_tags["critic_avg_value"]] = torch.mean(loss_q).item()
        tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000  # ms
        tb_info[tb_tags["loss_actor"]] = loss_policy.item()

        if self.per_flag:
            return tb_info, idx, abs_err
        else:
            return tb_info

    def __compute_loss_q(self, o, a, r, o2, d):
        q1 = self.networks.q1(o, a)
        q2 = self.networks.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = self.networks.policy_target(o2)
            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * self.target_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(
                a2,
                torch.tensor(self.act_low_limit).to(a2.device),
                torch.tensor(self.act_high_limit).to(a2.device),
            )

            # Target Q-values
            q1_pi_targ = self.networks.q1_target(o2, a2)
            q2_pi_targ = self.networks.q2_target(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q, loss_q1, loss_q2

    def __compute_loss_q_per(self, o, a, r, o2, d, idx, weight):
        q1 = self.networks.q1(o, a)
        q2 = self.networks.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = self.networks.policy_target(o2)
            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * self.target_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(
                a2,
                torch.tensor(self.act_low_limit).to(a2.device),
                torch.tensor(self.act_high_limit).to(a2.device),
            )

            # Target Q-values
            q1_pi_targ = self.networks.q1_target(o2, a2)
            q2_pi_targ = self.networks.q2_target(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = (weight * ((q1 - backup) ** 2)).mean()
        loss_q2 = (weight * ((q2 - backup) ** 2)).mean()
        loss_q = loss_q1 + loss_q2
        abs_err = torch.abs(q1 - backup)

        return loss_q, loss_q1, loss_q2, abs_err

    def __compute_loss_pi(self, o):
        q1_pi = self.networks.q1(o, self.networks.policy(o))
        return -q1_pi.mean()

    def __update(self, iteration):
        self.networks.q1_optimizer.step()
        self.networks.q2_optimizer.step()

        if iteration % self.delay_update == 0:
            self.networks.policy_optimizer.step()

        with torch.no_grad():
            polyak = 1 - self.tau
            for p, p_targ in zip(
                self.networks.q1.parameters(), self.networks.q1_target.parameters()
            ):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
            for p, p_targ in zip(
                self.networks.q2.parameters(), self.networks.q2_target.parameters()
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

        update_info = {
            "q1_grad": [p._grad for p in self.networks.q1.parameters()],
            "q2_grad": [p._grad for p in self.networks.q2.parameters()],
            "policy_grad": [p._grad for p in self.networks.policy.parameters()],
            "iteration": iteration,
        }

        return extra_info, update_info

    def remote_update(self, update_info: dict):
        iteration = update_info["iteration"]
        q1_grad = update_info["q1_grad"]
        q2_grad = update_info["q2_grad"]
        policy_grad = update_info["policy_grad"]

        for p, grad in zip(self.networks.q1.parameters(), q1_grad):
            p._grad = grad
        for p, grad in zip(self.networks.q2.parameters(), q2_grad):
            p._grad = grad
        for p, grad in zip(self.networks.policy.parameters(), policy_grad):
            p._grad = grad

        self.__update(iteration)
