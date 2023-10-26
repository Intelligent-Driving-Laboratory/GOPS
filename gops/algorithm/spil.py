#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Separated Proportional-Integral Lagrangian Algorithm
#  Reference: Peng B, Duan J, Chen J et al (2022) 
#             Model-Based Chance-Constrained Reinforcement Learning via Separated Proportional-Integral Lagrangian. 
#             IEEE Transactions on Neural Networks and Learning Systems. Doi: 10.1109/TNNLS.2022.3175595.
#  Update: 2021-03-05, Baiyu Peng: create SPIL algorithm


__all__ = ["SPIL"]

from copy import deepcopy
from typing import Any, Tuple

import torch
from torch.optim import Adam
import time
import numpy as np
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.create_pkg.create_env_model import create_env_model
from gops.utils.common_utils import get_apprfunc_dict
from gops.utils.tensorboard_setup import tb_tags
from gops.algorithm.base import AlgorithmBase, ApprBase


class ApproxContainer(ApprBase):
    """Approximate function container for SPIL.

    Contains one policy and one state value.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # create v network
        v_args = get_apprfunc_dict("value", **kwargs)
        self.v = create_apprfunc(**v_args)

        # create policy network
        policy_args = get_apprfunc_dict("policy", **kwargs)
        self.policy = create_apprfunc(**policy_args)

        # create target networks
        self.v_target = deepcopy(self.v)
        self.policy_target = deepcopy(self.policy)

        # set target networks gradients
        for p in self.v_target.parameters():
            p.requires_grad = False
        for p in self.policy_target.parameters():
            p.requires_grad = False

        # create optimizers
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=kwargs["policy_learning_rate"]
        )
        self.v_optimizer = Adam(self.v.parameters(), lr=kwargs["value_learning_rate"])

        self.net_dict = {"v": self.v, "policy": self.policy}
        self.target_net_dict = {"v": self.v_target, "policy": self.policy_target}
        self.optimizer_dict = {"v": self.v_optimizer, "policy": self.policy_optimizer}

    # create action_distributions
    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)


class SPIL(AlgorithmBase):
    """Separated Proportional-Integral Lagrangian (SPIL) algorithm

    Paper: https://ieeexplore.ieee.org/document/9785377

    :param float gamma: discount factor.
    :param float tau: param for soft update of target network.
    :param int pev_step: initial policy evaluation step.
    :param int pim_step: initial policy improvement step.
    :param int forward_step: predictive step in virtual horizon.
    """

    def __init__(
        self,
        index: int = 0,
        gamma: float = 0.99,
        tau: float = 0.005,
        pev_step: int = 1,
        pim_step: int = 1,
        forward_step: int = 25,
        **kwargs: Any,
    ):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.envmodel = create_env_model(**kwargs)
        self.gamma = gamma
        self.tau = tau
        self.pev_step = pev_step
        self.pim_step = pim_step
        self.forward_step = forward_step
        self.reward_scale = 1.0

        self.n_constraint = kwargs["constraint_dim"]
        self.delta_i = np.array([0.0] * kwargs["constraint_dim"])
        self.Kp = 60
        self.Ki = 0.02
        self.Kd = 0
        self.tb_info = dict()
        self.safe_prob_pre = np.array([0.0] * kwargs["constraint_dim"])
        self.chance_thre = np.array([0.97] * kwargs["constraint_dim"])

    @property
    def adjustable_parameters(self):
        para_tuple = (
            "gamma",
            "tau",
            "pev_step",
            "pim_step",
            "forward_step",
            "reward_scale",
        )
        return para_tuple

    def local_update(self, data: dict, iteration: int) -> dict:
        update_list = self.__compute_gradient(data, iteration)
        self.__update(update_list)
        return self.tb_info

    def get_remote_update_info(self, data: dict, iteration: int) -> Tuple[dict, dict]:
        update_list = self.__compute_gradient(data, iteration)
        update_info = dict()
        for net_name in update_list:
            update_info[net_name] = [
                p.grad for p in self.networks.net_dict[net_name].parameters()
            ]
        return self.tb_info, update_info

    def remote_update(self, update_info: dict):
        for net_name, grads in update_info.items():
            for p, grad in zip(self.networks.net_dict[net_name].parameters(), grads):
                p.grad = grad
        self.__update(list(update_info.keys()))

    def __update(self, update_list: list):
        tau = self.tau
        for net_name in update_list:
            self.networks.optimizer_dict[net_name].step()

        with torch.no_grad():
            for net_name in update_list:
                for p, p_targ in zip(
                    self.networks.net_dict[net_name].parameters(),
                    self.networks.target_net_dict[net_name].parameters(),
                ):
                    p_targ.data.mul_(1 - tau)
                    p_targ.data.add_(tau * p.data)

    def __compute_gradient(self, data: dict, iteration: int) -> list:
        update_list = []

        start_time = time.time()
        self.networks.v.zero_grad()
        loss_v, v = self.__compute_loss_v(deepcopy(data))
        loss_v.backward()
        self.tb_info[tb_tags["loss_critic"]] = loss_v.item()
        self.tb_info[tb_tags["critic_avg_value"]] = v.item()
        update_list.append("v")
        self.networks.policy.zero_grad()
        loss_policy = self.__compute_loss_policy(deepcopy(data))
        loss_policy.backward()
        self.tb_info[tb_tags["loss_actor"]] = loss_policy.item()
        update_list.append("policy")

        end_time = time.time()

        self.tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000

        return update_list

    def __compute_loss_v(self, data: dict):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        v = self.networks.v(o)
        traj_issafe = torch.ones(o.shape[0], self.n_constraint)
        info = data

        with torch.no_grad():
            for step in range(self.forward_step):
                if step == 0:
                    a = self.networks.policy(o)
                    o2, r, d, info = self.envmodel.forward(o, a, d, info)
                    r_sum = self.reward_scale * r
                    traj_issafe *= info["constraint"] <= 0

                else:
                    o = o2
                    a = self.networks.policy(o)
                    o2, r, d, info = self.envmodel.forward(o, a, d, info)
                    r_sum += self.reward_scale * self.gamma**step * r
                    traj_issafe *= info["constraint"] <= 0

            r_sum += self.gamma**self.forward_step * self.networks.v_target(o2)
        loss_v = ((v - r_sum) ** 2).mean()
        self.safe_prob = traj_issafe.mean(0).numpy()
        return loss_v, torch.mean(v)

    def __compute_loss_policy(self, data: dict):
        o, a, r, c, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["constraint"],
            data["obs2"],
            data["done"],
        )

        def Phi(y):
            # transfer constraint to cost
            m1 = 1
            m2 = m1 / (1 + m1) * 0.9
            tau = 0.07
            sig = (1 + tau * m1) / (
                1 + m2 * tau * torch.exp(torch.clamp(y / tau, min=-10, max=5))
            )
            return sig

        info = data
        for step in range(self.forward_step):
            if step == 0:
                a = self.networks.policy(o)
                o2, r, d, info = self.envmodel.forward(o, a, d, info)
                c = info["constraint"]
                c = Phi(c)
                r_sum = self.reward_scale * r
                c_sum = c
                c_mul = c
            else:
                o = o2
                a = self.networks.policy(o)
                o2, r, d, info = self.envmodel.forward(o, a, d, info)
                c = info["constraint"]
                c = Phi(c)
                r_sum = r_sum + self.reward_scale * self.gamma**step * r
                c_sum = c_sum + c
                c_mul = c_mul * c
        w_r, w_c = self.__spil_get_weight()
        loss_pi = (w_r * r_sum + (c_mul * torch.Tensor(w_c)).sum(1)).mean()
        return -loss_pi

    def __spil_get_weight(self):
        delta_p = self.chance_thre - self.safe_prob
        # integral separation
        delta_p_sepa = np.where(np.abs(delta_p) > 0.1, delta_p * 0.7, delta_p)
        delta_p_sepa = np.where(np.abs(delta_p) > 0.2, delta_p * 0, delta_p_sepa)
        self.delta_i = np.clip(self.delta_i + delta_p_sepa, 0, 99999)

        delta_d = np.clip(self.safe_prob_pre - self.safe_prob, 0, 3333)
        lam = np.clip(
            self.Ki * self.delta_i + self.Kp * delta_p + self.Kd * delta_d, 0, 3333
        )
        self.safe_prob_pre = self.safe_prob
        self.lam = lam
        return 1 / (1 + lam.sum()), lam / (1 + lam.sum())
