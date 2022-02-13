#   Copyright (c) 2020 ocp-tools Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Author: Sun Hao


__all__ = ['ApproxContainer','DDPG']

from copy import deepcopy
import torch
import torch.nn as nn
from torch.optim import Adam
import warnings
import time
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.utils import get_apprfunc_dict
from gops.utils.tensorboard_tools import tb_tags


class ApproxContainer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        value_func_type = kwargs['value_func_type']
        policy_func_type = kwargs['policy_func_type']

        if kwargs['cnn_shared']:  # todo:设置默认false
            feature_args = get_apprfunc_dict('feature', value_func_type, **kwargs)
            kwargs['feature_net'] = create_apprfunc(**feature_args)

        q_args = get_apprfunc_dict('value', value_func_type, **kwargs)
        self.q = create_apprfunc(**q_args)
        policy_args = get_apprfunc_dict('policy', policy_func_type, **kwargs)
        self.policy = create_apprfunc(**policy_args)

        self.q_target = deepcopy(self.q)
        self.policy_target = deepcopy(self.policy)

        for p in self.q_target.parameters():
            p.requires_grad = False
        for p in self.policy_target.parameters():
            p.requires_grad = False

        self.policy_optimizer = Adam(self.policy.parameters(), lr=kwargs['policy_learning_rate'])  # TODO:
        self.q_optimizer = Adam(self.q.parameters(), lr=kwargs['value_learning_rate'])

    # create action_distributions
    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)

    def update(self, grads_info:dict):
        iteration = grads_info['iteration']
        q_grad = grads_info['q_grad']
        policy_grad = grads_info['policy_grad']
        self.polyak = 1 - grads_info['tau']
        self.delay_update = grads_info['delay_update']

        for p, grad in zip(self.q.parameters(), q_grad):
            p._grad = grad
        for p, grad in zip(self.policy.parameters(), policy_grad):
            p._grad = grad
        self.q_optimizer.step()
        if iteration % self.delay_update == 0:
            self.policy_optimizer.step()
        with torch.no_grad():
            for p, p_targ in zip(self.q.parameters(), self.q_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(self.policy.parameters(), self.policy_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)


class DDPG:
    def __init__(self, **kwargs):
        self.networks = ApproxContainer(**kwargs)
        self.use_gpu = kwargs['use_gpu']
        self.gamma = 0.99
        self.tau = 0.005
        self.delay_update = 1
        self.reward_scale = 1

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
        params['tau'] = self.tau
        params['use_gpu'] = self.use_gpu
        params['delay_update'] = self.delay_update
        params['reward_scale'] = self.reward_scale
        return params

    def compute_gradient(self, data:dict, iteration):
        o, a, r, o2, d = data['obs'], data['act'], data['rew']*self.reward_scale, data['obs2'], data['done']
        # ------------------------------------*
        if self.use_gpu:
            self.networks.policy = self.networks.policy.cuda()
            self.networks.q = self.networks.q.cuda()
            self.networks.policy_target= self.networks.policy_target.cuda()
            self.networks.q_target = self.networks.q_target.cuda()
            o = o.cuda()
            a = a.cuda()
            r = r.cuda()
            o2 = o2.cuda()
            d = d.cuda()
        # ------------------------------------
        tb_info = dict()
        start_time = time.perf_counter()
        self.networks.q_optimizer.zero_grad()
        loss_q, q = self.__compute_loss_q( o, a, r, o2, d)
        loss_q.backward()

        for p in self.networks.q.parameters():
            p.requires_grad = False

        self.networks.policy_optimizer.zero_grad()
        loss_policy = self.__compute_loss_policy(o)
        loss_policy.backward()

        for p in self.networks.q.parameters():
            p.requires_grad = True
        #------------------------------------
        if self.use_gpu:
            self.networks.policy = self.networks.policy.cpu()
            self.networks.q = self.networks.q.cpu()
            self.networks.policy_target= self.networks.policy_target.cpu()
            self.networks.q_target = self.networks.q_target.cpu()
        # ------------------------------------
        q_grad = [p._grad for p in self.networks.q.parameters()]
        policy_grad = [p._grad for p in self.networks.policy.parameters()]

        # ------------------------------------
        end_time = time.perf_counter()
        tb_info[tb_tags["loss_critic"]] = loss_q.item()
        tb_info[tb_tags["critic_avg_value"]] = q.item()
        tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000  # ms
        tb_info[tb_tags["loss_actor"]] = loss_policy.item()
        # ------------------------------------
        grad_info = dict()
        grad_info['q_grad'] = q_grad
        grad_info['policy_grad'] = policy_grad
        grad_info['iteration'] = iteration
        grad_info['tau'] = self.tau
        grad_info['delay_update'] = self.delay_update

        return grad_info, tb_info

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)

    def __compute_loss_q(self,  o, a, r, o2, d):
        q = self.networks.q(o, a)

        q_policy_targ = self.networks.q_target(o2, self.networks.policy_target(o2))
        backup = r + self.gamma * (1 - d) * q_policy_targ

        loss_q = ((q - backup) ** 2).mean()
        return loss_q, torch.mean(q)

    def __compute_loss_policy(self, o):
        q_policy = self.networks.q(o, self.networks.policy(o))
        return -q_policy.mean()

    def update_policy(self, data:dict, iteration):
        grad_info, tb_info = self.compute_gradient(data, iteration)
        self.networks.update(grad_info)
        return grad_info, tb_info

    def state_dict(self,):
        return self.networks.state_dict()

if __name__ == '__main__':
    print('this is ddpg algorithm!')
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())