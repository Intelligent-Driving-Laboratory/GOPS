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
#  Update: 2021-03-05, Fawang Zhang: create FHADP algorithm
#  Update: 2022-12-04, Jiaxin Gao: supplementary comment information
#  Update: 2023-08-28, Guojian Zhan: support lr schedule

__all__ = ["FHADP"]

import time
from copy import deepcopy
from typing import Tuple

import torch
from torch.optim import Adam
from gops.algorithm.base import AlgorithmBase, ApprBase
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.create_pkg.create_env_model import create_env_model
from gops.utils.common_utils import get_apprfunc_dict
from gops.utils.gops_typing import DataDict, InfoDict
from gops.utils.tensorboard_setup import tb_tags


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

        self.policy = create_apprfunc(**policy_args)
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=policy_learning_rate
        )
        self.optimizer_dict = {
            "policy": self.policy_optimizer,
        }
        self.init_scheduler(**kwargs)

    def create_action_distributions(self, logits):
        """create action distribution"""
        return self.policy.get_act_dist(logits)


class FHADP(AlgorithmBase):
    """Approximate Dynamic Program Algorithm for Finity Horizon

    Paper: https://link.springer.com/book/10.1007/978-981-19-7784-8

    :param int pre_horizon: envmodel predict horizon.
    :param float gamma: discount factor.
    """

    def __init__(
        self,
        *,
        pre_horizon: int,
        gamma: float = 1.0,
        index: int = 0,
        **kwargs,
    ):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.envmodel = create_env_model(**kwargs, pre_horizon=pre_horizon)
        self.pre_horizon = pre_horizon
        self.gamma = gamma
        self.tb_info = dict()

    @property
    def adjustable_parameters(self) -> Tuple[str]:
        para_tuple = ("pre_horizon", "gamma")
        return para_tuple

    def _local_update(self, data: DataDict, iteration: int) -> InfoDict:
        self._compute_gradient(data)
        self.networks.policy_optimizer.step()
        return self.tb_info

    def get_remote_update_info(self, data: DataDict, iteration: int) -> Tuple[InfoDict, DataDict]:
        self._compute_gradient(data)
        policy_grad = [p._grad for p in self.networks.policy.parameters()]
        update_info = dict()
        update_info["grad"] = policy_grad
        return self.tb_info, update_info

    def _remote_update(self, update_info: DataDict):
        for p, grad in zip(self.networks.policy.parameters(), update_info["grad"]):
            p.grad = grad
        self.networks.policy_optimizer.step()

    def _compute_gradient(self, data: DataDict):
        start_time = time.time()
        self.networks.policy.zero_grad()
        loss_policy, loss_info = self._compute_loss_policy(deepcopy(data))
        loss_policy.backward()
        end_time = time.time()
        self.tb_info.update(loss_info)
        self.tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000  # ms

    def _compute_loss_policy(self, data: DataDict) -> Tuple[torch.Tensor, InfoDict]:
        o, d = data["obs"], data["done"]
        info = data
        v_pi = 0
        for step in range(self.pre_horizon):
            a = self.networks.policy(o, step + 1)
            o, r, d, info = self.envmodel.forward(o, a, d, info)
            v_pi += r * (self.gamma ** step)
        loss_policy = -v_pi.mean()
        loss_info = {
            tb_tags["loss_actor"]: loss_policy.item()
        }
        return loss_policy, loss_info
