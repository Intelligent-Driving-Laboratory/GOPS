#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Finity ADP Algorithm
#  Update: 2022-09-17, Jie Li: create relaxed PI algorithm


__all__ = ['ApproxContainer', 'RPI']

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
from gops.utils.act_distribution import DiracDistribution
from gops.utils.tensorboard_setup import tb_tags
from gops.algorithm.base import AlgorithmBase, ApprBase


class ApproxContainer(ApprBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.env_model = create_env_model(**kwargs)
        self.learning_rate = kwargs['learning_rate']

        value_func_type = kwargs['value_func_type']
        value_args = get_apprfunc_dict('value', value_func_type, **kwargs)
        self.value = create_apprfunc(**value_args)
        self.gt_weight = kwargs.get('gt_weight', None)
        initial_weight = kwargs.get('initial_weight', None)
        if initial_weight is not None:
            # weight initialization
            self.v.weight = Parameter(torch.tensor(initial_weight, dtype=torch.float32), requires_grad=True)
        else:
            if value_func_type == 'POLY':
                # zero initialization
                self.value.v.weight.data.fill_(0)
            else:
                for m in self.value.v:
                    if isinstance(m, nn.Linear):
                        weight_shape = list(m.weight.data.size())
                        fan_in = weight_shape[1]
                        fan_out = weight_shape[0]
                        w_bound = np.sqrt(6. / (fan_in + fan_out))
                        m.weight.data.uniform_(-w_bound, w_bound)
                        m.bias.data.fill_(0)
        self.value_target = deepcopy(self.value)

    def policy(self, batch_obs):
        batch_obs.requires_grad_(True)
        batch_value_target = self.value_target(batch_obs)
        batch_delta_value_target, = torch.autograd.grad(torch.sum(batch_value_target), batch_obs, create_graph=True)
        batch_obs.requires_grad_(False)
        batch_act = self.env_model.best_act(batch_obs.detach(), batch_delta_value_target)
        return batch_act

    def action_and_adversary(self, batch_obs):
        batch_obs.requires_grad_(True)
        batch_value_target = self.value_target(batch_obs)
        batch_delta_value_target, = torch.autograd.grad(torch.sum(batch_value_target), batch_obs, create_graph=True)
        batch_obs.requires_grad_(False)
        batch_act = self.env_model.best_act(batch_obs.detach(), batch_delta_value_target)
        batch_adv = self.env_model.worst_adv(batch_obs.detach(), batch_delta_value_target)
        return torch.cat((batch_act, batch_adv), dim=1)

    @staticmethod
    def create_action_distributions(logits):
        return DiracDistribution(logits)


class RPI(AlgorithmBase):
    def __init__(self, **kwargs):
        super().__init__(index=0, **kwargs)
        self.value_func_type = kwargs['value_func_type']
        self.max_newton_iteration = kwargs['max_newton_iteration']
        self.max_step_update_value = kwargs['max_step_update_value']
        self.print_interval = kwargs['print_interval']
        self.obsv_dim = kwargs['obsv_dim']
        self.act_dim = kwargs['action_dim']
        self.gt_weight = kwargs.get('gt_weight', None)

        self.num_update_value = 0
        self.norm_hamiltonian_before = 0
        self.norm_hamiltonian_after = self.max_step_update_value ** 3
        self.step_size_newton = 0
        self.set_state = None
        self.grad_step = np.ones([int(self.max_newton_iteration), 1], dtype="float32")

        self.env_id = kwargs['env_id']
        self.is_adversary = kwargs.get('is_adversary', False)
        self.env_model = create_env_model(**kwargs)
        self.obs = self.env_model.reset()
        self.done = None
        self.env_model.unwrapped.parallel_state = self.obs.clone()

        self.networks = ApproxContainer(**kwargs)
        self.learning_rate = kwargs["learning_rate"]
        self.approximate_optimizer = Adam(self.networks.parameters(), lr=self.learning_rate, betas=(0.9, 0.99), weight_decay=0)

    def continue_evaluation(self):
        return abs(self.norm_hamiltonian_after) > 0.88 * abs(self.norm_hamiltonian_before) \
               and self.num_update_value < self.max_step_update_value

    @property
    def adjustable_parameters(self):
        return (
            "max_newton_iteration",
        )

    def local_update(self, data_useless, iteration):
        self.num_update_value = 0
        self.set_state = self.env_model.reset().clone()
        self.norm_hamiltonian_before = self.__calculate_norm_hamiltonian(self.set_state)

        start_time = time.time()

        for i in range(self.max_step_update_value):
            data = self.sample()

            self.approximate_optimizer.zero_grad()
            loss_value = self.__compute_loss(data)
            loss_value.backward()
            self.approximate_optimizer.step()

            self.norm_hamiltonian_after = self.__calculate_norm_hamiltonian(self.set_state)
            self.num_update_value += 1

            if not self.continue_evaluation():
                break

        self.networks.value_target = deepcopy(self.networks.value)
        end_time = time.time()

        grad_info = dict()
        grad_info['iteration'] = iteration
        grad_info['num_update_value'] = self.num_update_value
        grad_info[tb_tags["loss_critic"]] = loss_value.item()
        grad_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000  # ms

        if iteration % self.print_interval == 0:
            self.grad_step[iteration, 0] = self.num_update_value
            print(f'Newton ite: {iteration}, grad step = {self.num_update_value:d}, '
                  f'loss value = {math.log10(loss_value.item()):.2f}')
            if self.value_func_type == 'POLY' or self.value_func_type == 'POLYNOMIAL':
                weight = self.networks.value.v.weight.detach()[0]
                print(f'weight = {weight}')
                if self.gt_weight is not None:
                    gt = np.array(self.gt_weight)
                    print(f'error = {math.log10(np.linalg.norm(weight - gt) / np.linalg.norm(gt)):.2f}')

        return grad_info

    def __compute_loss(self, data):
        obs, act, adv = data['obs'], data['act'], data['advers']

        # name completion
        batch_observation = obs
        batch_input = torch.cat((act, adv), dim=1)

        # value loss
        batch_observation.requires_grad_(True)
        batch_value = self.networks.value(batch_observation)
        batch_delta_value, = torch.autograd.grad(torch.sum(batch_value), batch_observation, create_graph=True)
        batch_observation.requires_grad_(False)
        done = torch.zeros(obs.shape[0], device=obs.device).bool()
        info = {}
        _, batch_utility, _, next_info = self.env_model.forward(batch_observation, batch_input, done, info)
        batch_delta_state = next_info['delta_state']
        loss_value = self.__value_loss_function(batch_delta_value, batch_utility.detach(), batch_delta_state.detach())

        return loss_value

    def __calculate_norm_hamiltonian(self, set_state):
        # name completion
        batch_observation = set_state
        batch_input = self.networks.action_and_adversary(set_state)

        # hamiltonian
        batch_observation.requires_grad_(True)
        batch_value = self.networks.value(batch_observation)
        batch_delta_value, = torch.autograd.grad(torch.sum(batch_value), batch_observation, create_graph=True)
        batch_observation.requires_grad_(False)
        done = torch.zeros(set_state.shape[0], device=set_state.device).bool()
        info = {}
        _, batch_utility, _, next_info = self.env_model.forward(batch_observation, batch_input, done, info)
        batch_delta_state = next_info['delta_state']
        hamiltonian = self.__value_loss_function(batch_delta_value, batch_utility.detach(), batch_delta_state.detach())

        return hamiltonian.detach().item()

    @staticmethod
    def __value_loss_function(delta_value, utility, delta_state):
        # \partial V / \partial t * f（x, u ,w）= dV / dt
        dv_dt = torch.diag(torch.mm(delta_value, delta_state.t()), 0)  # take the main diagonal as a 1-dim tensor
        # dv_dt = torch.bmm(delta_value[:, np.newaxis, :], delta_state[:, :, np.newaxis]).squeeze(-1).squeeze(-1)
        hamiltonian = utility + dv_dt
        loss = torch.mean(torch.abs(hamiltonian))
        return loss

    def sample(self):
        action = self.networks.action_and_adversary(self.obs)
        next_obs, reward, self.done, info = self.env_model.step(action)
        if 'TimeLimit.truncated' not in info.keys():
            info['TimeLimit.truncated'] = self.env_model.zeros_.bool()
        # self.done = torch.where(info['TimeLimit.truncated'], self.env_model.zeros_, self.done).bool()

        data_dict = {}
        data_dict.update({'obs': self.obs.clone(),
                          'act': action[:, :self.act_dim],
                          'rew': reward,
                          'obs2': next_obs.clone(),
                          'done': self.done,
                          'time_limited': info['TimeLimit.truncated']})
        if self.is_adversary:
            data_dict.update({'advers': action[:, self.act_dim:]})
        else:
            data_dict.update({'advers': None})
        self.obs = next_obs

        reset_obs = self.env_model.reset()
        reset_signal = self.done | info['TimeLimit.truncated']
        self.obs = torch.where(reset_signal.unsqueeze(dim=-1).repeat(1, self.obsv_dim),
                               reset_obs, self.obs)
        self.env_model.unwrapped.parallel_state = self.obs.clone()
        self.env_model.step_per_episode = torch.where(self.done | info['TimeLimit.truncated'],
                                                      self.env_model.initial_step(), self.env_model.step_per_episode)

        return data_dict

