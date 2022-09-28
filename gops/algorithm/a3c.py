#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Asynchronous Advantage Actor Critic Algorithm (A3C)
#  Update: 2021-03-05, Jiaxin Gao: create A3C algorithm


__all__ = ["ApproxContainer", "A3C"]

from typing import Tuple
from gops.algorithm.base import AlgorithmBase, ApprBase
from torch.optim import Adam
import time
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.common_utils import get_apprfunc_dict
from gops.utils.tensorboard_setup import tb_tags


class ApproxContainer(ApprBase):
    """Approximate function container for A3C.

    Contains a policy and a state value.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        value_func_type = kwargs["value_func_type"]
        policy_func_type = kwargs["policy_func_type"]

        v_args = get_apprfunc_dict("value", value_func_type, **kwargs)
        self.v = create_apprfunc(**v_args)
        policy_args = get_apprfunc_dict("policy", policy_func_type, **kwargs)
        self.policy = create_apprfunc(**policy_args)

        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=kwargs["policy_learning_rate"]
        )
        self.v_optimizer = Adam(
            self.v.parameters(), lr=kwargs["value_learning_rate"]
        )

    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)


class A3C(AlgorithmBase):
    """Asynchronous Advantage Actor Critic (A3C) algorithm

    Paper: https://arxiv.org/abs/1602.01783
    
    """
    
    def __init__(self, index=0, **kwargs):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.gamma = 0.99
        self.reward_scale = 1
        self.delay_update = 1

    @property
    def adjustable_parameters(self):
        return (
            "gamma",
            "reward_scale",
            "delay_update"
        )


    def __compute_gradient(self, data: dict, iteration):
        o, a, r, o2 = (
            data["obs"],
            data["act"],
            data["rew"] * self.reward_scale,
            data["obs2"],
        )

        tb_info = dict()
        start_time = time.time()

        self.networks.v_optimizer.zero_grad()
        loss_v, v = self.__compute_loss_v(o, r, o2)
        loss_v.backward()

        for p in self.networks.v.parameters():
            p.requires_grad = False

        self.networks.policy_optimizer.zero_grad()
        loss_policy = self.__compute_loss_policy(o, a, r, o2)
        loss_policy.backward()

        for p in self.networks.v.parameters():
            p.requires_grad = True

        # ------------------------------------
        end_time = time.time()
        tb_info[tb_tags["loss_critic"]] = loss_v.item()
        tb_info[tb_tags["critic_avg_value"]] = v.item()
        tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000  # ms
        tb_info[tb_tags["loss_actor"]] = loss_policy.item()

        return tb_info

    def __compute_loss_v(self, o, r, o2):
        v = self.networks.v(o)
        target_v = r + self.gamma * self.networks.v(o2)
        loss_v = ((v - target_v) ** 2).mean()

        return loss_v, v.detach().mean()

    def __compute_loss_policy(self, o, a, r, o2):
        # one_step advantage r + gamma * V(obs2) - V(obs)        
        logits = self.networks.policy(o)
        action_distribution = self.networks.create_action_distributions(logits)
        logp = action_distribution.log_prob(a)
        v_policy = logp * (
            r + self.gamma * self.networks.v(o2) - self.networks.v(o)
        )
        
        return -v_policy.mean()

    def __update(self, iteration):
        self.networks.v_optimizer.step()
        if iteration % self.delay_update == 0:
            self.networks.policy_optimizer.step()

    def local_update(self, data: dict, iteration: int):
        tb_info = self.__compute_gradient(data, iteration)
        self.__update(iteration)
        return tb_info

    def get_remote_update_info(self, data: dict, iteration: int) -> Tuple[dict, dict]:
        tb_info = self.__compute_gradient(data, iteration)

        v_grad = [p._grad for p in self.networks.v.parameters()]
        policy_grad = [p._grad for p in self.networks.policy.parameters()]

        update_info = dict()
        update_info["v_grad"] = v_grad
        update_info["policy_grad"] = policy_grad
        update_info["iteration"] = iteration

        return tb_info, update_info

    def remote_update(self, update_info: dict):
        iteration = update_info["iteration"]
        v_grad = update_info["v_grad"]
        policy_grad = update_info["policy_grad"]

        for p, grad in zip(self.networks.v.parameters(), v_grad):
            p._grad = grad
        for p, grad in zip(self.networks.policy.parameters(), policy_grad):
            p._grad = grad

        self.__update(iteration)
