#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Deep Q-Learning Algorithm (DQN)
#  Update: 2021-03-05, Wenxuan Wang: create DQN algorithm


__all__ = ["ApproxContainer", "DQN"]


from copy import deepcopy
import time
import warnings
from typing import Dict
import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import Tuple

from gops.algorithm.base import AlgorithmBase, ApprBase
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.common_utils import get_apprfunc_dict
from gops.utils.tensorboard_setup import tb_tags

class ApproxContainer(ApprBase):
    def __init__(self, **kwargs):
        """Approximate function container for DQN.

        Contains an action value.
        """
        super().__init__(**kwargs)
        value_func_type = kwargs["value_func_type"]

        q_args = get_apprfunc_dict("value", value_func_type, **kwargs)
        self.q = create_apprfunc(**q_args)

        self.q_target = deepcopy(self.q)

        for p in self.q_target.parameters():
            p.requires_grad = False

        # the policy directly comes from the Q func, and is just for sampling
        def policy_q(obs):
            with torch.no_grad():
                return self.q.forward(obs)
        self.policy = policy_q

        self.q_optimizer = Adam(self.q.parameters(), lr=kwargs["value_learning_rate"])

    def create_action_distributions(self, logits):
        return self.q.get_act_dist(logits)

class DQN(AlgorithmBase):
    """Deep Q-Network (DQN) algorithm

            A DQN implementation with soft target update.

            Paper: https://doi.org/10.1038/nature14236

            Args:
                learning_rate (float, optional): Q network learning rate. Defaults to 0.001.
                gamma (float, optional): Discount factor. Defaults to 0.995.
                tau (float, optional): Average factor. Defaults to 0.005.
            """
    def __init__(self, index=0, **kwargs):
        super().__init__(index, **kwargs)
        self.gamma = 0.99
        self.tau = 0.005
        self.reward_scale = 1
        self.networks = ApproxContainer(**kwargs)
        self.per_flag = (kwargs["buffer_name"] == "prioritized_replay_buffer")

    @property
    def adjustable_parameters(self):
        return (
            "gamma", 
            "tau", 
            "reward_scale"
        )

    def set_parameters(self, param_dict):
        for key in param_dict:
            if hasattr(self, key):
                setattr(self, key, param_dict[key])
            else:
                warning_msg = "param '" + key + "'is not defined in algorithm!"
                warnings.warn(warning_msg)

    def __compute_gradient(self, data: Dict[str, torch.Tensor], iteration: int):
        tb_info = dict()
        start_time = time.perf_counter()

        self.networks.q_optimizer.zero_grad()
        if not self.per_flag:
            o, a, r, o2, d = (
                data["obs"],
                data["act"],
                data["rew"] * self.reward_scale,
                data["obs2"],
                data["done"],
            )
            loss_q = self.__compute_loss_q(o, a, r, o2, d)
            loss_q.backward()
        else:
            o, a, r, o2, d, idx, weight = (
                data["obs"],
                data["act"],
                data["rew"] * self.reward_scale,
                data["obs2"],
                data["done"],
                data["idx"],
                data["weight"]
            )
            loss_q, abs_err = self.__compute_loss_per(o, a, r, o2, d, idx, weight)
            loss_q.backward()

        end_time = time.perf_counter()

        tb_info[tb_tags["loss_critic"]] = loss_q.item()
        tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000  # ms

        if self.per_flag:
            return tb_info, idx, abs_err
        else:
            return tb_info

    def __compute_loss_q(self, o, a, r, o2, d):
        q = self.networks.q(o).gather(1, a.to(torch.long)).squeeze()

        with torch.no_grad():
            q_target, _ = torch.max(self.networks.q_target(o2), dim=1)
        backup = r + self.gamma * (1 - d) * q_target

        loss_q = F.mse_loss(q, backup)
        return loss_q

    def __compute_loss_per(self, o, a, r, o2, d, idx, weight):
        q = self.networks.q(o).gather(1, a.to(torch.long)).squeeze()

        with torch.no_grad():
            q_target, _ = torch.max(self.networks.target(o2), dim=1)
        backup = r + self.gamma * (1 - d) * q_target

        loss_q = (weight * ((q - backup) ** 2)).mean()
        abs_err = torch.abs(q - backup)
        return loss_q, abs_err

    def __update(self, iteration):
        polyak = 1 - self.tau

        self.networks.q_optimizer.step()

        with torch.no_grad():
            for p, p_targ in zip(self.networks.q.parameters(), self.networks.q_target.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def local_update(self, data: dict, iteration: int):
        extra_info = self.__compute_gradient(data, iteration)
        self.__update(iteration)
        return extra_info

    def get_remote_update_info(self, data: dict, iteration: int) -> Tuple[dict, dict]:
        extra_info = self.__compute_gradient(data, iteration)

        q_grad = [p._grad for p in self.networks.q.parameters()]

        update_info = dict()
        update_info["q_grad"] = q_grad
        update_info["iteration"] = iteration

        return extra_info, update_info

    def remote_update(self, update_info: dict):
        iteration = update_info["iteration"]
        q_grad = update_info["q_grad"]

        for p, grad in zip(self.networks.q.parameters(), q_grad):
            p._grad = grad

        self.__update(iteration)
