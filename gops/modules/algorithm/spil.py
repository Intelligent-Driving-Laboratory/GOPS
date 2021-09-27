#   Copyright (c) 2020 ocp-tools Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Author: Sun Hao
#  Update Date: 2020-11-13
#  Update Date: 2021-01-03
#  Comments: ?


__all__ = ['SPIL']

from copy import deepcopy
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import time
import warnings

from modules.create_pkg.create_apprfunc import create_apprfunc
from modules.create_pkg.create_env_model import create_env_model
from modules.utils.utils import get_apprfunc_dict
from modules.utils.tensorboard_tools import tb_tags

#需要修改的地方。
# 1.环境中定义好一个constrained dim
# 2.buffer，sampler，evaluator做判断，如果存在这个维度需要能够接受算法返回的con

class ApproxContainer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # self.polyak = 1 - kwargs['tau']
        value_func_type = kwargs['value_func_type']
        policy_func_type = kwargs['policy_func_type']
        v_args = get_apprfunc_dict('value', value_func_type, **kwargs)
        policy_args = get_apprfunc_dict('policy', policy_func_type, **kwargs)

        self.v = create_apprfunc(**v_args)
        self.policy = create_apprfunc(**policy_args)

        self.v_target = deepcopy(self.v)
        self.policy_target = deepcopy(self.policy)

        for p in self.v_target.parameters():
            p.requires_grad = False
        for p in self.policy_target.parameters():
            p.requires_grad = False

        self.policy_optimizer = Adam(self.policy.parameters(), lr=kwargs['policy_learning_rate'])  #
        self.v_optimizer = Adam(self.v.parameters(), lr=kwargs['value_learning_rate'])

        self.net_dict = {'v': self.v, 'policy': self.policy}
        self.target_net_dict = {'v': self.v_target, 'policy': self.policy_target}
        self.optimizer_dict = {'v': self.v_optimizer, 'policy': self.policy_optimizer}

    def update(self, grad_info):
        tau = grad_info['tau']
        grads_dict = grad_info['grads_dict']
        for net_name, grads in grads_dict.items():
            for p, grad in zip(self.net_dict[net_name].parameters(), grads):
                p.grad = grad
            self.optimizer_dict[net_name].step()

        with torch.no_grad():
            for net_name in grads_dict.keys():
                for p, p_targ in zip(self.net_dict[net_name].parameters(), self.target_net_dict[net_name].parameters()):
                    p_targ.data.mul_(1-tau)
                    p_targ.data.add_(tau * p.data)

class SPIL:
    def __init__(self, **kwargs):
        self.networks = ApproxContainer(**kwargs)
        self.envmodel = create_env_model(**kwargs)
        self.use_gpu = kwargs['use_gpu']
        if self.use_gpu:
            self.envmodel = self.envmodel.cuda()
        self.gamma = 0.99
        self.tau = 0.005
        self.pev_step = 1
        self.pim_step = 1
        self.forward_step = 25
        self.reward_scale = 0.02

        self.n_constraint = kwargs['constraint_dim']
        self.delta_i = np.array([0.] * kwargs['constraint_dim'])

        self.Kp = 40
        self.Ki = 0.07*5
        self.Kd = 0

        self.tb_info = dict()

        self.safe_prob_pre = np.array([0.] * kwargs['constraint_dim'])
        self.chance_thre = np.array([0.99] * kwargs['constraint_dim'])

    def set_parameters(self, param_dict):
        for key in param_dict:
            if hasattr(self, key):
                setattr(self, key, param_dict[key])
            else:
                warning_msg = "param '" + key + "'is not defined in algorithm!"
                warnings.warn(warning_msg)

    def get_parameters(self):
        params = dict()
        params['use_gpu'] = self.use_gpu
        params['gamma'] = self.gamma
        params['tau'] = self.tau
        params['pev_step'] = self.pev_step
        params['pim_step'] = self.pim_step
        params['reward_scale'] = self.reward_scale
        params['forward_step'] = self.forward_step
        return params

    def compute_gradient(self, data, iteration):
        grad_info = dict()
        grads_dict = dict()

        start_time = time.time()
        if self.use_gpu:
            self.networks = self.networks.cuda()
            for key, value in data.items():
                data[key] = value.cuda()

        # if iteration % (self.pev_step + self.pim_step) < self.pev_step: ##TODO: 这里改成了每个iteration都包含pev和pim
        self.networks.v.zero_grad()
        loss_v, v = self.compute_loss_v(deepcopy(data))
        loss_v.backward()
        v_grad = [p.grad for p in self.networks.v.parameters()]
        self.tb_info[tb_tags["loss_critic"]] = loss_v.item()
        self.tb_info[tb_tags["critic_avg_value"]] = v.item()
        grads_dict['v'] = v_grad
        # else:
        self.networks.policy.zero_grad()
        loss_policy = self.compute_loss_policy(deepcopy(data))
        loss_policy.backward()
        policy_grad = [p.grad for p in self.networks.policy.parameters()]
        self.tb_info[tb_tags["loss_actor"]] = loss_policy.item()
        grads_dict['policy'] = policy_grad

        if self.use_gpu:
            self.networks = self.networks.cpu()
            for key, value in data.items():
                data[key] = value.cpu()

        end_time = time.time()


        self.tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000  # ms
        self.tb_info[tb_tags["safe_probability1"]] = self.safe_prob[0].item()
        self.tb_info[tb_tags["lambda1"]] = self.lam[0].item()
        self.tb_info[tb_tags["safe_probability2"]] = self.safe_prob[1].item()
        self.tb_info[tb_tags["lambda2"]] = self.lam[1].item()

        # writer.add_scalar(tb_tags['Lambda'], self.lam, iter)
        # writer.add_scalar(tb_tags['Safe_prob'], self.safe_prob, iter)

        grad_info['tau'] = self.tau
        grad_info['grads_dict'] = grads_dict
        return grad_info, self.tb_info

        # tb_info[tb_tags["loss_critic"]] = loss_v.item()
        # tb_info[tb_tags["critic_avg_value"]] = v.item()
        # tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000  # ms
        # tb_info[tb_tags["loss_actor"]] = loss_policy.item()
        # return v_grad + policy_grad, tb_info

    def compute_loss_v(self, data):
        o, a, r, c, o2, d = data['obs'], data['act'], data['rew'], data['con'], data['obs2'], data['done']
        v = self.networks.v(o)
        traj_issafe = torch.ones(o.shape[0], self.n_constraint)

        with torch.no_grad():
            for step in range(self.forward_step):
                if step == 0:
                    a = self.networks.policy(o)
                    o2, r, d, _ = self.envmodel.forward(o, a, d)
                    r_sum = self.reward_scale * r
                    traj_issafe *= (~d)

                else:
                    o = o2
                    a = self.networks.policy(o)
                    o2, r, d, _ = self.envmodel.forward(o, a, d)
                    r_sum += self.reward_scale * self.gamma ** step * r
                    traj_issafe *= (~d)

            r_sum += self.gamma ** self.forward_step * self.networks.v_target(o2)
        loss_v = ((v - r_sum) ** 2).mean()
        self.safe_prob = (traj_issafe).mean(0).numpy()
        print(r_sum.mean(), self.safe_prob)
        return loss_v, torch.mean(v)

    def compute_loss_policy(self, data):
        o, a, r, c, o2, d = data['obs'], data['act'], data['rew'], data['con'], data['obs2'], data['done']  # TODO  解耦字典
        for step in range(self.forward_step):
            if step == 0:
                a = self.networks.policy(o)
                o2, r, d, info = self.envmodel.forward(o, a, d)
                c = info['constraint']
                c = self.Phi(c)
                r_sum = self.reward_scale * r
                c_sum = c
                c_mul = c
            else:
                o = o2
                a = self.networks.policy(o)
                o2, r, d, info = self.envmodel.forward(o, a, d)
                c = info['constraint']
                c = self.Phi(c)
                r_sum = r_sum + self.reward_scale * self.gamma ** step * r
                c_sum = c_sum + c
                c_mul = c_mul * c
        #r_sum += self.gamma ** self.forward_step * self.networks.v_target(o2)
        w_r, w_c = self.spil_get_weight()
        loss_pi = (w_r * r_sum + (c_mul*torch.Tensor(w_c)).sum(1)).mean()
        return -loss_pi

    def Phi(self, y):
        # Transfer constraint to cost
        m1 = 1
        m2 = m1 / (1 + m1) * 0.9
        tau = 0.07
        sig = (1 + tau * m1) / (1 + m2 * tau * torch.exp(torch.clamp(y / tau, min=-10, max=5)))
        # c = torch.relu(-y)
        return sig

        # The following is for max
        # m1 = 3/2
        # m2 = m1 / (1 + m1) * 1
        # m2 = 3/2
        # tau = 0.2
        # sig = (1 + tau * m1) / (1 + m2 * tau * torch.exp(torch.clamp(y / tau, min=-5, max=5)))

    def spil_get_weight(self):
        delta_p = (self.chance_thre - self.safe_prob)
        # integral separation
        delta_p_sepa = np.where(np.abs(delta_p) > 0.1, delta_p * 0.7, delta_p)
        delta_p_sepa = np.where(np.abs(delta_p) > 0.2, delta_p * 0, delta_p_sepa)
        self.delta_i = np.clip(self.delta_i + delta_p_sepa, 0, 99999)

        delta_d = np.clip(self.safe_prob_pre - self.safe_prob, 0, 3333)
        lam = np.clip(self.Ki * self.delta_i + self.Kp * delta_p + self.Kd * delta_d, 0, 3333)
        self.safe_prob_pre = self.safe_prob
        self.lam = lam
        self.tb_info[tb_tags["I1"]] = self.delta_i[0].item()
        self.tb_info[tb_tags["I2"]] = self.delta_i[1].item()
        return 1 / (1+lam.sum()), lam / (1+lam.sum())
        #return 1, lam / (1 + lam.sum())


    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)


if __name__ == '__main__':
    print('11111')
