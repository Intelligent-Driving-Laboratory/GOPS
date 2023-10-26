#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Approximate Dynamic Program Algorithm for Finity Horizon (FHADP)
#  Reference: Li SE (2023)
#             Reinforcement Learning for Sequential Decision and Optimal Control. Springer, Singapore.
#  create: 2023-07-28, Jiaxin Gao: create full horizon action fhadp algorithm

__all__ = ["FHADP2"]

from copy import deepcopy
from typing import Tuple
import torch
import torch.nn as nn
from torch.optim import Adam
import time
import warnings
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.create_pkg.create_env_model import create_env_model
from gops.utils.common_utils import get_apprfunc_dict
from gops.utils.tensorboard_setup import tb_tags
from gops.algorithm.base import AlgorithmBase, ApprBase


class ApproxContainer(ApprBase):
    def __init__(self, **kwargs):
        """Approximate function container for FHADP."""
        """Contains one policy network."""
        super().__init__(**kwargs)
        policy_args = get_apprfunc_dict("policy", **kwargs)

        self.policy = create_apprfunc(**policy_args)
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=kwargs["policy_learning_rate"]
        )

    def create_action_distributions(self, logits):
        """create action distribution"""
        return self.policy.get_act_dist(logits)


class FHADP2(AlgorithmBase):
    """Approximate Dynamic Program Algorithm for Finity Horizon

    Paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4124940

    :param int forward_step: envmodel forward step.
    :param float gamma: discount factor.
    """

    def __init__(self, index=0, **kwargs):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.envmodel = create_env_model(**kwargs)
        self.forward_step = kwargs["pre_horizon"]
        self.gamma = 1.0
        self.tb_info = dict()

    @property
    def adjustable_parameters(self):
        para_tuple = ("forward_step", "gamma")
        return para_tuple

    def local_update(self, data, iteration: int):
        self.__compute_gradient(data)
        self.networks.policy_optimizer.step()
        return self.tb_info

    def get_remote_update_info(self, data: dict, iteration: int) -> Tuple[dict, dict]:
        self.__compute_gradient(data)
        policy_grad = [p._grad for p in self.networks.policy.parameters()]
        update_info = dict()
        update_info["grad"] = policy_grad
        return self.tb_info, update_info

    def remote_update(self, update_info: dict):
        for p, grad in zip(self.networks.policy.parameters(), update_info["grad"]):
            p.grad = grad
        self.networks.policy_optimizer.step()

    def __compute_gradient(self, data):
        start_time = time.time()
        self.networks.policy.zero_grad()
        loss_policy = self.__compute_loss_policy(deepcopy(data))

        loss_policy.backward()

        self.tb_info[tb_tags["loss_actor"]] = loss_policy.item()

        end_time = time.time()

        self.tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000  # ms

        return

    def __compute_loss_policy(self, data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        info_init = data
        v_pi = 0
        a = self.networks.policy.forward_all_policy(o)
        for step in range(self.forward_step):
            if step == 0:
                o2, r, d, info = self.envmodel.forward(o, a[:, 0, :], d, info_init)
                v_pi = r
            else:
                o = o2
                o2, r, d, info = self.envmodel.forward(o, a[:, step, :], d, info)
                v_pi += r * (self.gamma**step)

        return -(v_pi).mean()


if __name__ == "__main__":
    print("11111")
