#   Copyright (c) 2020 ocp-tools Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Author: Jiaxin Gao
# A3C algorithm need cooperate with asynchronous or synchronous trainer

__all__ = ['ApproxContainer', 'A3C']

from copy import deepcopy
import torch
import torch.nn as nn
from torch.optim import Adam
import warnings
import time
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.utils import get_apprfunc_dict
from gops.utils.tensorboard_tools import tb_tags
from gops.utils.action_distributions import GaussDistribution


class ApproxContainer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        value_func_type = kwargs['value_func_type']
        policy_func_type = kwargs['policy_func_type']

        if kwargs['cnn_shared']:  # todo:设置默认false
            feature_args = get_apprfunc_dict('feature', value_func_type, **kwargs)
            kwargs['feature_net'] = create_apprfunc(**feature_args)


        value_args = get_apprfunc_dict('value', value_func_type, **kwargs)
        self.value = create_apprfunc(**value_args)
        policy_args = get_apprfunc_dict('policy', policy_func_type, **kwargs)
        self.policy = create_apprfunc(**policy_args)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=kwargs['policy_learning_rate'])
        self.value_optimizer = Adam(self.value.parameters(), lr=kwargs['value_learning_rate'])

    # create action_distributions
    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)

    def update(self, grads_info: dict):
        iteration = grads_info['iteration']
        value_grad = grads_info['value_grad']
        policy_grad = grads_info['policy_grad']
        self.delay_update = grads_info['delay_update']
        # self.polyak = 1 - grads_info['tau']
        # self.delay_update = grads_info['delay_update']

        for p, grad in zip(self.value.parameters(), value_grad):
            p._grad = grad
        for p, grad in zip(self.policy.parameters(), policy_grad):
            p._grad = grad
        self.value_optimizer.step()
        if iteration % self.delay_update == 0:
            self.policy_optimizer.step()


class A3C:
    def __init__(self, **kwargs):
        self.networks = ApproxContainer(**kwargs)
        self.use_gpu = kwargs['use_gpu']
        self.gamma = 0.99
        self.reward_scale = 1
        self.delay_update =  1
        self.action_distirbution_cls = GaussDistribution

    def set_parameters(self, param_dict):
        for key in param_dict:
            if hasattr(self, key):
                setattr(self, key, param_dict[key])
            else:
                warning_msg = "param '" + key + "'is not defined in algorithm!"
                warnings.warn(warning_msg)

    def get_parameters(self):
        params = dict()
        params['gamma'] = self.gamma
        params['use_gpu'] = self.use_gpu
        params['reward_scale'] = self.reward_scale
        params['delay_update'] = self.delay_update
        return params

    def compute_gradient(self, data: dict, iteration):
        # o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        # ------------------------------------
        # if self.use_gpu:
        #     self.networks.policy = self.networks.policy.cuda()
        #     self.networks.value = self.networks.value.cuda()
        #     o = o.cuda()
        #     a = a.cuda()
        #     r = r.cuda()
        #     o2 = o2.cuda()
        #     d = d.cuda()
        # ------------------------------------
        tb_info = dict()
        start_time = time.time()

        self.networks.value_optimizer.zero_grad()
        loss_value, value = self.__compute_loss_value(data)
        # print('loss_value = ', loss_value)
        loss_value.backward()

        for p in self.networks.value.parameters():
            p.requires_grad = False

        self.networks.policy_optimizer.zero_grad()
        loss_policy = self.__compute_loss_policy(data)
        # print('loss_policy = ', loss_policy)
        loss_policy.backward()

        for p in self.networks.value.parameters():
            p.requires_grad = True

        #------------------------------------
        # if self.use_gpu:
        #     self.networks.policy = self.networks.policy.cpu()
        #     self.networks.value = self.networks.value.cpu()
        # ------------------------------------
        value_grad = [p._grad for p in self.networks.value.parameters()]
        policy_grad = [p._grad for p in self.networks.policy.parameters()]

        # ------------------------------------
        end_time = time.time()
        tb_info[tb_tags["loss_critic"]] = loss_value.item()
        tb_info[tb_tags["critic_avg_value"]] = value.item()
        tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000  # ms
        tb_info[tb_tags["loss_actor"]] = loss_policy.item()
        # ------------------------------------
        grad_info = dict()
        grad_info['value_grad'] = value_grad
        grad_info['policy_grad'] = policy_grad
        grad_info['iteration'] = iteration
        grad_info['delay_update'] = self.delay_update

        return grad_info, tb_info

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)

    def __compute_loss_value(self, data):
        obs = data['obs']
        rew = data['rew']*self.reward_scale
        done = data['done']
        obs2 = data['obs2']
        value = self.networks.value(obs)
        target_value = rew + self.gamma * self.networks.value(obs2)
        # target_value = torch.where(done == 1, rew, rew + self.gamma * self.networks.value(obs2))
        loss_value = ((value - target_value) ** 2).mean()
        return loss_value, value.detach().mean()

    def __compute_loss_policy(self, data):
        # one_step advantage r + V(obs2) - V(obs)
        obs = data['obs']
        rew = data['rew']*self.reward_scale
        done = data['done']
        obs2 = data['obs2']
        action = data['act']
        logits = self.networks.policy(obs)
        action_distribution = self.action_distirbution_cls(logits)
        logp = action_distribution.log_prob(action)
        value_policy = logp * (rew + self.gamma * self.networks.value(obs2) - self.networks.value(obs))
        # value_policy = torch.where(done == 1, logp * (rew - self.networks.value(obs)), logp * (rew + self.gamma * self.networks.value(obs2) - self.networks.value(obs)))
        return -value_policy.mean()

    def update_policy(self, data: dict, iteration):
        grad_info, tb_info = self.compute_gradient(data, iteration)
        self.networks.update(grad_info)
        return grad_info, tb_info

    def state_dict(self,):
        return self.networks.state_dict()
