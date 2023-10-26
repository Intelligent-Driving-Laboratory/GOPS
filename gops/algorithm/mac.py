#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Mixed Actor Critic Algorithm (MAC)
#  Reference: Mu Y, Li SE, Liu C et al (2020) 
#             Mixed reinforcement learning for efficient policy optimization in stochastic environments. 
#             IEEE ICCAS, Pusan, Korea.
#  Update: 2021-03-05, Yao Mu: create MAC algorithm


__all__ = ["MAC"]

from copy import deepcopy
from typing import Tuple

import torch
from torch.optim import Adam
import time

from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.create_pkg.create_env_model import create_env_model
from gops.utils.common_utils import get_apprfunc_dict
from gops.utils.tensorboard_setup import tb_tags
from gops.algorithm.base import AlgorithmBase, ApprBase
import numpy as np


class ApproxContainer(ApprBase):
    def __init__(self, **kwargs):
        """Approximate function container for DDPG.

        Contains a policy and an action value.
        """
        super().__init__(**kwargs)

        v_args = get_apprfunc_dict("value", **kwargs)
        policy_args = get_apprfunc_dict("policy", **kwargs)

        self.v = create_apprfunc(**v_args)
        self.policy = create_apprfunc(**policy_args)

        self.v_target = deepcopy(self.v)
        self.policy_target = deepcopy(self.policy)

        for p in self.v_target.parameters():
            p.requires_grad = False
        for p in self.policy_target.parameters():
            p.requires_grad = False

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

    def update(self, grad_info):
        tau = grad_info["tau"]
        grads_dict = grad_info["grads_dict"]
        for net_name, grads in grads_dict.items():
            for p, grad in zip(self.net_dict[net_name].parameters(), grads):
                p.grad = grad
            self.optimizer_dict[net_name].step()

        with torch.no_grad():
            for net_name in grads_dict.keys():
                for p, p_targ in zip(
                    self.net_dict[net_name].parameters(),
                    self.target_net_dict[net_name].parameters(),
                ):
                    p_targ.data.mul_(1 - tau)
                    p_targ.data.add_(tau * p.data)


class MAC(AlgorithmBase):
    """Mixed Actor Critic Algorithm (MAC) algorithm

    Paper:https://ieeexplore.ieee.org/document/9268413

    :param float gamma: discount factor.
    :param float tau: param for soft update of target network.
    :param int pev_step: number of steps for policy evaluation.
    :param int pim_step: number of steps for policy improvement.
    :param int forward_step: envmodel forward step.
    """

    def __init__(self, index=0, **kwargs):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.envmodel = create_env_model(**kwargs)
        self.use_gpu = kwargs["use_gpu"]
        if self.use_gpu:
            self.envmodel = self.envmodel.cuda()
        self.gamma = 0.99
        self.tau = 0.005
        self.pev_step = 1
        self.pim_step = 1
        self.forward_step = 10
        self.reward_scale = 1
        self.tb_info = dict()
        self.delta = None

    @property
    def adjustable_parameters(self):
        para_tuple = (
            "gamma",
            "tau",
            "pev_step",
            "pim_step",
            "forward_step",
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

    def __update(self, update_list):
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

    def dynamic_model_forward(self, o, a, d):
        if self.delta is not None:
            self.delta = torch.zeros_like(o)
        o2, r, d, _ = self.envmodel.forward(o, a, d, {})
        o2 = o2 + self.delta
        return o2, r, d

    def update_ibe_model(self, o, a, d, o2):
        data = o2 - self.envmodel.forward(o, a, d, {})[0]
        zero_prior_mean = torch.zeros_like(data[0])
        diag_variance = 0.5 * torch.ones_like(torch.diag(data[0]))
        self.delta = self.iterative_bayes_estimator(
            data, zero_prior_mean, diag_variance
        )

    def iterative_bayes_estimator(self, data, basic_mu, basic_var):
        N, input_dim = data.shape
        var = torch.diag(torch.var(data, 0))
        data_sum = torch.sum(data, 0).unsqueeze(1)
        basic_mu = basic_mu.unsqueeze(1)

        for i in range(4):
            K = torch.pinverse(torch.pinverse(basic_var) + N * torch.pinverse(var))
            Z = torch.mm(torch.pinverse(basic_var), basic_mu) + torch.mm(
                torch.pinverse(var), data_sum
            )
            mu = torch.mm(K, Z)
            var = torch.mm((data - mu.t()).t(), data - mu.t()) / N
        mu = mu.detach().cpu().numpy()
        var = var.detach().cpu().numpy()
        sample = torch.tensor(
            np.random.multivariate_normal(np.squeeze(mu), var, N), dtype=torch.float32
        ).detach()
        if self.use_gpu:
            sample = sample.cuda()
        return sample

    def __compute_gradient(self, data, iteration):
        update_list = []

        start_time = time.perf_counter()

        if iteration % (self.pev_step + self.pim_step) < self.pev_step:
            self.networks.v.zero_grad()
            loss_v, v = self.compute_loss_v(deepcopy(data))
            loss_v.backward()
            self.tb_info[tb_tags["loss_critic"]] = loss_v.item()
            self.tb_info[tb_tags["critic_avg_value"]] = v.item()
            update_list.append("v")
        else:
            self.networks.policy.zero_grad()
            loss_policy = self.compute_loss_policy(deepcopy(data))
            loss_policy.backward()
            self.tb_info[tb_tags["loss_actor"]] = loss_policy.item()
            update_list.append("policy")

        end_time = time.perf_counter()

        self.tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000  # ms

        return update_list

    def compute_loss_v(self, data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        self.update_ibe_model(o, a, d, o2)
        v = self.networks.v(o)
        with torch.no_grad():
            for step in range(self.forward_step):
                if step == 0:
                    a = self.networks.policy(o)
                    o2, r, d = self.dynamic_model_forward(o, a, d)
                    backup = self.reward_scale * r
                else:
                    o = o2
                    a = self.networks.policy(o)
                    o2, r, d = self.dynamic_model_forward(o, a, d)
                    backup += self.reward_scale * self.gamma**step * r

            backup += (
                (~d) * self.gamma**self.forward_step * self.networks.v_target(o2)
            )
        loss_v = ((v - backup) ** 2).mean()
        return loss_v, torch.mean(v)

    def compute_loss_policy(self, data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        v_pi = torch.zeros(1)
        for p in self.networks.v.parameters():
            p.requires_grad = False
        for step in range(self.forward_step):
            if step == 0:
                a = self.networks.policy(o)
                o2, r, d = self.dynamic_model_forward(o, a, d)
                v_pi = self.reward_scale * r
            else:
                o = o2
                a = self.networks.policy(o)
                o2, r, d = self.dynamic_model_forward(o, a, d)
                v_pi += self.reward_scale * self.gamma**step * r
        v_pi += (~d) * self.gamma**self.forward_step * self.networks.v_target(o2)
        for p in self.networks.v.parameters():
            p.requires_grad = True
        return -v_pi.mean()
