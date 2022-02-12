#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Sun Hao
#  Description: Soft Actor-Critic
#
#  Update Date: 2021-6-17, Yang Yujie: implement SAC
#
#  Supported environment: gym_cartpoleconti, gym_pendulum
#  Supported trainer: off_serial, off_async, on_serial, on_sync
#  (Note that on-policy trainers are not recommended since SAC is an off-policy algorithm)

__all__ = ['ApproxContainer', 'SAC']

import time
import warnings
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import Adam

from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.action_distributions import GaussDistribution
from gops.utils.tensorboard_tools import tb_tags
from gops.utils.utils import get_apprfunc_dict


class ApproxContainer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # create value network
        value_func_type = kwargs['value_func_type']
        value_args = get_apprfunc_dict('value', value_func_type, **kwargs)

        if kwargs['cnn_shared']:  # todo:设置默认false
            feature_args = get_apprfunc_dict('feature', value_func_type, **kwargs)
            kwargs['feature_net'] = create_apprfunc(**feature_args)

        self.value = create_apprfunc(**value_args)

        # create q networks
        q_func_type = kwargs['q_func_type']
        q_args = get_apprfunc_dict('q', q_func_type, **kwargs)
        self.q1 = create_apprfunc(**q_args)
        self.q2 = create_apprfunc(**q_args)

        # create policy network
        policy_func_type = kwargs['policy_func_type']
        policy_args = get_apprfunc_dict('policy', policy_func_type, **kwargs)
        self.policy = create_apprfunc(**policy_args)

        # create target network
        self.value_target = deepcopy(self.value)

        # set target network gradients
        for p in self.value_target.parameters():
            p.requires_grad = False

        # create optimizers
        self.value_optimizer = Adam(self.value.parameters(), lr=kwargs['value_learning_rate'])
        self.q1_optimizer = Adam(self.q1.parameters(), lr=kwargs['q_learning_rate'])
        self.q2_optimizer = Adam(self.q2.parameters(), lr=kwargs['q_learning_rate'])
        self.policy_optimizer = Adam(self.policy.parameters(), lr=kwargs['policy_learning_rate'])

    # create action_distributions
    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)

    def update(self, grads_info):
        value_grad = grads_info['value_grad']
        q1_grad = grads_info['q1_grad']
        q2_grad = grads_info['q2_grad']
        policy_grad = grads_info['policy_grad']
        polyak = 1 - grads_info['tau']

        # update value network
        for p, grad in zip(self.value.parameters(), value_grad):
            p._grad = grad
        self.value_optimizer.step()

        # update q networks
        for p, grad in zip(self.q1.parameters(), q1_grad):
            p._grad = grad
        for p, grad in zip(self.q2.parameters(), q2_grad):
            p._grad = grad
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # update policy network
        for p, grad in zip(self.policy.parameters(), policy_grad):
            p._grad = grad
        self.policy_optimizer.step()

        # update target network
        with torch.no_grad():
            for p, p_targ in zip(self.value.parameters(), self.value_target.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)


class SAC:
    def __init__(self, **kwargs):
        self.networks = ApproxContainer(**kwargs)
        self.act_dist_cls = GaussDistribution
        self.use_gpu = kwargs['use_gpu']
        self.gamma = 0.99
        self.tau = 0.005
        self.auto_alpha = True
        self.target_entropy = -kwargs['action_dim']

        if self.auto_alpha:
            self.log_alpha = torch.tensor(0, dtype=torch.float32)
            if self.use_gpu:
                self.log_alpha = self.log_alpha.cuda()
            self.log_alpha.requires_grad = True
            self.alpha_optimizer = Adam([self.log_alpha], lr=kwargs['alpha_learning_rate'])
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = 0.2

    def set_parameters(self, param_dict):
        for key in param_dict:
            if hasattr(self, key):
                setattr(self, key, param_dict[key])
            else:
                warning_msg = "param '" + key + "' is not defined in algorithm!"
                warnings.warn(warning_msg)

    def get_parameters(self):
        params = dict()
        params['gamma'] = self.gamma
        params['tau'] = self.tau
        params['use_gpu'] = self.use_gpu
        params['auto_alpha'] = self.auto_alpha
        params['alpha'] = self.alpha
        params['target_entropy'] = self.target_entropy
        return params

    def compute_gradient(self, data, iteration):
        start_time = time.time()

        if self.use_gpu:
            self.networks = self.networks.cuda()
            for k, v in data.items():
                data[k] = v.cuda()

        obs = data['obs']
        logits = self.networks.policy(obs)
        act_dist = self.networks.create_action_distributions(logits)
        new_act = act_dist.rsample()
        new_logp = act_dist.log_prob(new_act)
        data.update({
            'new_act': new_act,
            'new_logp': new_logp
        })

        self.networks.value_optimizer.zero_grad()
        loss_value, value = self._compute_loss_value(data)
        loss_value.backward()

        self.networks.q1_optimizer.zero_grad()
        self.networks.q2_optimizer.zero_grad()
        loss_q, q1, q2 = self._compute_loss_q(data)
        loss_q.backward()

        for p in self.networks.q1.parameters():
            p.requires_grad = False
        for p in self.networks.q2.parameters():
            p.requires_grad = False

        self.networks.policy_optimizer.zero_grad()
        loss_policy, entropy = self._compute_loss_policy(data)
        loss_policy.backward()

        for p in self.networks.q1.parameters():
            p.requires_grad = True
        for p in self.networks.q2.parameters():
            p.requires_grad = True

        if self.auto_alpha:
            self.alpha_optimizer.zero_grad()
            loss_alpha = self._compute_loss_alpha(data)
            loss_alpha.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        if self.use_gpu:
            self.networks = self.networks.cpu()

        grad_info = {
            'value_grad': [p.grad for p in self.networks.value.parameters()],
            'q1_grad': [p.grad for p in self.networks.q1.parameters()],
            'q2_grad': [p.grad for p in self.networks.q2.parameters()],
            'policy_grad': [p.grad for p in self.networks.policy.parameters()],
            'tau': self.tau
        }

        tb_info = {
            tb_tags['loss_critic']: loss_value.item(),
            tb_tags['loss_actor']: loss_policy.item(),
            tb_tags['critic_avg_value']: value.item(),
            'Train/critic_avg_q1': q1.item(),
            'Train/critic_avg_q2': q2.item(),
            'Train/entropy': entropy.item(),
            'Train/alpha': self.alpha,
            tb_tags['alg_time']: (time.time() - start_time) * 1000
        }

        return grad_info, tb_info

    def _compute_loss_value(self, data):
        obs, new_act, new_logp = data['obs'], data['new_act'], data['new_logp']
        value = self.networks.value(obs).squeeze(-1)
        with torch.no_grad():
            q1 = self.networks.q1(obs, new_act)
            q2 = self.networks.q2(obs, new_act)
            target_value = torch.min(q1, q2) - self.alpha * new_logp
        loss_value = ((value - target_value) ** 2).mean()
        return loss_value, value.detach().mean()

    def _compute_loss_q(self, data):
        obs, act, rew, obs2, done = \
            data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q1 = self.networks.q1(obs, act)
        q2 = self.networks.q2(obs, act)
        with torch.no_grad():
            next_value = self.networks.value_target(obs2).squeeze(-1)
            backup = rew + (1 - done) * self.gamma * next_value
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        return loss_q1 + loss_q2, q1.detach().mean(), q2.detach().mean()

    def _compute_loss_policy(self, data):
        obs, new_act, new_logp = data['obs'], data['new_act'], data['new_logp']
        q1 = self.networks.q1(obs, new_act)
        q2 = self.networks.q2(obs, new_act)
        loss_policy = (self.alpha * new_logp - torch.min(q1, q2)).mean()
        entropy = -new_logp.detach().mean()
        return loss_policy, entropy

    def _compute_loss_alpha(self, data):
        new_logp = data['new_logp']
        loss_alpha = -self.log_alpha.exp() * (new_logp.detach() + self.target_entropy).mean()
        return loss_alpha

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)
