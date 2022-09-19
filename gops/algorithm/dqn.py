#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Deep Q-Learning Algorithm (DQN)
#  Update: 2021-03-05, Wenxuan Wang: create DQN algorithm


__all__ = ["DQN"]


from copy import deepcopy
import time
import warnings
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Tuple

from gops.algorithm.base import AlgorithmBase, ApprBase
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.common_utils import get_apprfunc_dict
from gops.utils.tensorboard_setup import tb_tags


class ApproxContainer(ApprBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        value_func_type = kwargs["value_func_type"]
        Q_network_dict = get_apprfunc_dict("value", value_func_type, **kwargs)
        Q_network: nn.Module = create_apprfunc(**Q_network_dict)
        target_network = deepcopy(Q_network)
        target_network.eval()
        for p in target_network.parameters():
            p.requires_grad = False

        def policy_q(obs):
            with torch.no_grad():
                return self.q.forward(obs)

        self.policy = policy_q
        self.q = Q_network
        self.target = target_network
        self.q_optimizer = Adam(self.q.parameters(), lr=kwargs["value_learning_rate"])

    def create_action_distributions(self, logits):
        return self.q.get_act_dist(logits)

    def update(self, grads_info: dict):
        polyak = 1 - grads_info["tau"]
        q_grad = grads_info["q_grad"]
        for p, grad in zip(self.q.parameters(), q_grad):
            p._grad = grad
        self.q_optimizer.step()

        with torch.no_grad():
            for p, p_targ in zip(self.q.parameters(), self.target.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)


class DQN(AlgorithmBase):
    def __init__(self, index=0, **kwargs):
        """Deep Q-Network (DQN) algorithm

        A DQN implementation with soft target update.

        Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control through deep reinforcement learning.
        Nature 518, 529~533 (2015). https://doi.org/10.1038/nature14236

        Args:
            learning_rate (float, optional): Q network learning rate. Defaults to 0.001.
            gamma (float, optional): Discount factor. Defaults to 0.995.
            tau (float, optional): Average factor. Defaults to 0.005.
        """
        super().__init__(index, **kwargs)
        self.gamma = 0.99
        self.use_gpu = kwargs["use_gpu"]
        self.tau = 0.005
        self.reward_scale = 1
        self.networks = ApproxContainer(**kwargs)

    @property
    def adjustable_parameters(self):
        para_tuple = ("gamma", "tau", "reward_scale")
        return para_tuple

    def set_parameters(self, param_dict):
        for key in param_dict:
            if hasattr(self, key):
                setattr(self, key, param_dict[key])
            else:
                warning_msg = "param '" + key + "'is not defined in algorithm!"
                warnings.warn(warning_msg)

    def get_parameters(self):
        params = dict()
        params["gamma"] = self.gamma
        params["tau"] = self.tau
        params["use_gpu"] = self.use_gpu
        params["reward_scale"] = self.reward_scale
        return params

    def __compute_gradient(self, data: Dict[str, torch.Tensor], iteration: int):
        start_time = time.perf_counter()
        obs, act, rew, obs2, done = (
            data["obs"],
            data["act"],
            data["rew"] * self.reward_scale,
            data["obs2"],
            data["done"],
        )
        if self.use_gpu:
            self.networks.cuda()
            obs, act, rew, obs2, done = (
                obs.cuda(),
                act.cuda(),
                rew.cuda(),
                obs2.cuda(),
                done.cuda(),
            )

        self.networks.q_optimizer.zero_grad()
        loss = self.compute_loss(obs, act, rew, obs2, done)
        loss.backward()

        if self.use_gpu:
            self.networks.cpu()

        end_time = time.perf_counter()

        # q_grad = [p._grad for p in self.networks.q.parameters()]
        tb_info = {
            tb_tags["loss_critic"]: loss.item(),
            tb_tags["alg_time"]: (end_time - start_time) * 1000,
        }

        # grad_info = dict()
        # grad_info["q_grad"] = q_grad
        # grad_info["tau"] = self.tau

        return tb_info

    def __update(self, iteration):
        self.networks.q_optimizer.step()

        with torch.no_grad():
            polyak = 1 - self.tau
            for p, p_targ in zip(self.networks.q.parameters(), self.networks.target.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def local_update(self, data: dict, iteration: int):
        tb_info = self.__compute_gradient(data, iteration)
        self.__update(iteration)
        return tb_info

    def get_remote_update_info(self, data: dict, iteration: int) -> Tuple[dict, dict]:
        tb_info = self.__compute_gradient(data, iteration)

        update_info = {
            "q_grad": [p._grad for p in self.networks.q.parameters()],
            "iteration": iteration,
        }

        return tb_info, update_info

    def remote_update(self, update_info: dict):
        iteration = update_info["iteration"]
        q_grad = update_info["q_grad"]

        for p, grad in zip(self.networks.q.parameters(), q_grad):
            p._grad = grad

        self.__update(iteration)

    def compute_loss(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        obs2: torch.Tensor,
        done: torch.Tensor,
    ):
        q_policy = self.networks.q(obs).gather(1, act.to(torch.long)).squeeze()

        with torch.no_grad():
            q_target, _ = torch.max(self.networks.target(obs2), dim=1)
        q_expect = rew + self.gamma * (1 - done) * q_target

        loss = F.mse_loss(q_policy, q_expect)
        return loss

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.networks.load_state_dict(state_dict)
