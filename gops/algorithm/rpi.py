#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Relaxed Policy Iteration (RPI) Algorithm
#  Reference: Li J, Li SE, Guan Y et al (2020) 
#             Ternary Policy Iteration Algorithm for Nonlinear Robust Control. 
#             https://arxiv.org/abs/2007.06810.
#  Update Date: 2022-09-17, Jie Li: create RPI algorithm


__all__ = ["ApproxContainer", "RPI"]

from copy import deepcopy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.parameter import Parameter
import time
import warnings

from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.create_pkg.create_env_model import create_env_model
from gops.utils.common_utils import get_apprfunc_dict
from gops.utils.act_distribution_cls import Action_Distribution
from gops.utils.act_distribution_type import DiracDistribution
from gops.utils.tensorboard_setup import tb_tags
from gops.algorithm.base import AlgorithmBase, ApprBase


class ApproxContainer(ApprBase):
    """Approximate function container for RPI.
    Args:
        str value_func_type: type of value network.
        list initial_weight: initial weight of value network.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.env_model = create_env_model(**kwargs)

        # create value network
        value_args = get_apprfunc_dict("value", **kwargs)
        self.value = create_apprfunc(**value_args)

        # initialize value network
        initial_weight = kwargs.get("initial_weight", None)
        if initial_weight is not None:
            # weight initialization
            self.v.weight = Parameter(
                torch.tensor(initial_weight, dtype=torch.float32), requires_grad=True
            )
        else:
            if kwargs["value_func_type"] == "POLY":
                # zero initialization
                self.value.v.weight.data.fill_(0)
            else:
                for m in self.value.v:
                    if isinstance(m, nn.Linear):
                        weight_shape = list(m.weight.data.size())
                        fan_in = weight_shape[1]
                        fan_out = weight_shape[0]
                        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
                        m.weight.data.uniform_(-w_bound, w_bound)
                        m.bias.data.fill_(0)

        # create target network
        self.value_target = deepcopy(self.value)

    # create policy function
    def policy(self, batch_obs):
        batch_obs.requires_grad_(True)
        batch_value_target = self.value_target(batch_obs)
        (batch_delta_value_target,) = torch.autograd.grad(
            torch.sum(batch_value_target), batch_obs, create_graph=True
        )
        batch_obs.requires_grad_(False)
        batch_act = self.env_model.best_act(
            batch_obs.detach(), batch_delta_value_target
        )
        return batch_act

    # joint function of action and adversary
    def action_and_adversary(self, batch_obs):
        batch_obs.requires_grad_(True)
        batch_value_target = self.value_target(batch_obs)
        (batch_delta_value_target,) = torch.autograd.grad(
            torch.sum(batch_value_target), batch_obs, create_graph=True
        )
        batch_obs.requires_grad_(False)
        batch_act = self.env_model.best_act(
            batch_obs.detach(), batch_delta_value_target
        )
        batch_adv = self.env_model.worst_adv(
            batch_obs.detach(), batch_delta_value_target
        )
        return torch.cat((batch_act, batch_adv), dim=1)

    # create action_distributions
    @staticmethod
    def create_action_distributions(logits):
        return DiracDistribution(logits)


class RPI(AlgorithmBase):
    """
    Relaxed Policy Iteration (RPI) algorithm
    Paper: https://arxiv.org/abs/2007.06810.
    """

    def __init__(
        self,
        index: int = 0,
        max_newton_iteration: int = 50,
        max_step_update_value: int = 10000,
        print_interval: int = 1,
        learning_rate: float = 1e-3,
        **kwargs,
    ) -> None:
        """
        Relaxed Policy Iteration (RPI) algorithm.
            :param: int max_newton_iteration: max iteration in Newton's method.
            :param: int max_step_update_value: max gradient step in policy evaluation.
            :param: int print_interval: print interval.
            :param: float learning_rate: learning rate of value function.
        """
        super().__init__(index, **kwargs)

        self.max_newton_iteration = max_newton_iteration
        self.max_step_update_value = max_step_update_value
        self.print_interval = print_interval

        self.num_update_value = 0
        self.norm_hamiltonian_before = 0
        self.norm_hamiltonian_after = self.max_step_update_value**3
        self.step_size_newton = 0
        self.set_state = None
        self.grad_step = np.ones([int(self.max_newton_iteration), 1], dtype="float32")

        self.is_adversary = kwargs['is_adversary']
        self.env_model = create_env_model(**kwargs)
        self.obsv_dim = self.env_model.state_dim
        self.act_dim = self.env_model.action_dim
        self.obs = self.env_model.reset()
        self.done = None
        self.env_model.unwrapped.parallel_state = self.obs.clone()

        self.networks = ApproxContainer(**kwargs)
        self.learning_rate = learning_rate
        self.approximate_optimizer = Adam(
            self.networks.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.99),
            weight_decay=0,
        )

    # terminal condition for policy evaluation
    def continue_evaluation(self):
        return (
            abs(self.norm_hamiltonian_after) > 0.88 * abs(self.norm_hamiltonian_before)
            and self.num_update_value < self.max_step_update_value
        )

    @property
    def adjustable_parameters(self):
        return ("max_newton_iteration",)

    def local_update(self, data_useless, iteration):
        self.num_update_value = 0
        start_time = time.time()

        # threshold value to determine whether to continue policy evaluation
        self.set_state = self.env_model.reset().clone()
        self.norm_hamiltonian_before = self.__calculate_norm_hamiltonian(self.set_state)

        # policy evaluation and update value network
        for i in range(self.max_step_update_value):
            self.num_update_value += 1
            data = self.sample()

            self.approximate_optimizer.zero_grad()
            loss_value = self.__compute_loss(data)
            loss_value.backward()
            self.approximate_optimizer.step()

            # judge whether to continue policy evaluation
            self.norm_hamiltonian_after = self.__calculate_norm_hamiltonian(
                self.set_state
            )
            if not self.continue_evaluation():
                break

        # update target value network
        self.networks.value_target = deepcopy(self.networks.value)
        end_time = time.time()

        # log information
        grad_info = dict()
        grad_info["iteration"] = iteration
        grad_info["num_update_value"] = self.num_update_value
        grad_info[tb_tags["loss_critic"]] = loss_value.item()
        grad_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000  # ms

        # print information
        if iteration % self.print_interval == 0:
            self.grad_step[iteration, 0] = self.num_update_value
            print(f"Newton ite: {iteration}, grad step = {self.num_update_value:d}")

        return grad_info

    # compute value loss
    def __compute_loss(self, data):
        # get data including state, action, and adversary
        obs, act, adv = data["obs"], data["act"], data["advers"]

        # name completion
        batch_observation = obs
        batch_input = torch.cat((act, adv), dim=1)

        # value loss
        loss_value = self.__calculate_hamiltonian(batch_observation, batch_input)

        return loss_value

    # for policy evaluation terminal condition
    def __calculate_norm_hamiltonian(self, set_state):
        # name completion
        batch_observation = set_state
        batch_input = self.networks.action_and_adversary(set_state)

        # hamiltonian
        hamiltonian = self.__calculate_hamiltonian(batch_observation, batch_input)

        return hamiltonian.detach().item()

    def __calculate_hamiltonian(self, batch_observation, batch_input):
        """
        calculate Hamiltonian.
        :param: torch.tensor batch_observation: state.
        :param: torch.tensor batch_input: action.
        :return: torch.tensor hamiltonian: Hamiltonian of state and action pair.
        """
        batch_observation.requires_grad_(True)
        batch_value = self.networks.value(batch_observation)
        (batch_delta_value,) = torch.autograd.grad(
            torch.sum(batch_value), batch_observation, create_graph=True
        )
        batch_observation.requires_grad_(False)

        done = torch.zeros(
            batch_observation.shape[0], device=batch_observation.device
        ).bool()
        info = {}
        _, batch_reward, _, next_info = self.env_model.forward(
            batch_observation, batch_input, done, info
        )

        batch_utility = -batch_reward
        batch_delta_state = next_info["delta_state"]
        hamiltonian = self.__value_loss_function(
            batch_delta_value, batch_utility.detach(), batch_delta_state.detach()
        )

        return hamiltonian

    @staticmethod
    def __value_loss_function(delta_value, utility, delta_state):
        """
        value loss for continuous-time zero-sum game.
        :param: torch.tensor delta_value: partial value partial state.
        :param: torch.tensor utility: cost function.
        :param: torch.tensor delta_state: system dynamics.
        :return: torch.tensor loss: value loss.
        """
        # dV / dt = \partial V / \partial t * f(x, u, w)
        dv_dt = torch.diag(torch.mm(delta_value, delta_state.t()), 0)
        # hamiltonian
        hamiltonian = utility + dv_dt
        # value loss
        loss = torch.mean(torch.abs(hamiltonian))
        return loss

    def sample(self):
        # rollout
        action = self.networks.action_and_adversary(self.obs)
        next_obs, reward, self.done, info = self.env_model.step(action)
        if "TimeLimit.truncated" not in info.keys():
            info["TimeLimit.truncated"] = self.env_model.zeros_.bool()

        # collect data including state, action, reward, next state, done, time_limited and adversary
        data_dict = {}
        data_dict.update(
            {
                "obs": self.obs.clone(),
                "act": action[:, : self.act_dim],
                "rew": reward,
                "obs2": next_obs.clone(),
                "done": self.done,
                "time_limited": info["TimeLimit.truncated"],
            }
        )
        if self.is_adversary:
            data_dict.update({"advers": action[:, self.act_dim :]})
        else:
            data_dict.update({"advers": None})
        self.obs = next_obs

        # reset some agents
        reset_obs = self.env_model.reset()
        reset_signal = self.done | info["TimeLimit.truncated"]
        self.obs = torch.where(
            reset_signal.unsqueeze(dim=-1).repeat(1, self.obsv_dim), reset_obs, self.obs
        )
        self.env_model.unwrapped.parallel_state = self.obs.clone()
        self.env_model.step_per_episode = torch.where(
            self.done | info["TimeLimit.truncated"],
            self.env_model.initial_step(),
            self.env_model.step_per_episode,
        )

        return data_dict
