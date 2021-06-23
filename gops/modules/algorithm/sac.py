#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Sun Hao
#  Description: Soft Actor-Critic
#
#  Update Date: 2021-6-17, Yang Yujie: implement SAC

__all__ = ['SAC']

import time
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import Adam

from modules.create_pkg.create_apprfunc import create_apprfunc
from modules.utils.action_distributions import GaussDistribution
from modules.utils.tensorboard_tools import tb_tags
from modules.utils.utils import get_apprfunc_dict


class ApproxContainer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.polyak = 1 - kwargs['tau']

        self.log_alpha = nn.Parameter(torch.tensor(kwargs['alpha'], dtype=torch.float32).log())
        self.target_entropy = -kwargs['action_dim']

        # create value network
        value_func_type = kwargs['value_func_type']
        value_args = get_apprfunc_dict('value', value_func_type, **kwargs)
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
        self.alpha_optimizer = Adam([self.log_alpha], lr=kwargs['alpha_learning_rate'])

    def update(self, grads, iteration):
        value_grad_len = len(list(self.value.parameters()))
        q_grad_len = len(list(self.q1.parameters()))
        value_grad = grads[:value_grad_len]
        q1_grad = grads[value_grad_len:value_grad_len + q_grad_len]
        q2_grad = grads[value_grad_len + q_grad_len:value_grad_len + 2 * q_grad_len]
        policy_grad = grads[value_grad_len + 2 * q_grad_len:-1]
        alpha_grad = grads[-1]

        # update value network
        for p, grad in zip(self.value.parameters(), value_grad):
            p._grad = torch.from_numpy(grad)
        self.value_optimizer.step()

        # update q networks
        for p, grad in zip(self.q1.parameters(), q1_grad):
            p._grad = torch.from_numpy(grad)
        for p, grad in zip(self.q2.parameters(), q2_grad):
            p._grad = torch.from_numpy(grad)
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # update policy network
        for p, grad in zip(self.policy.parameters(), policy_grad):
            p._grad = torch.from_numpy(grad)
        self.policy_optimizer.step()

        # update target network
        with torch.no_grad():
            for p, p_targ in zip(self.value.parameters(), self.value_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        # update alpha
        self.log_alpha._grad = torch.from_numpy(alpha_grad)
        self.alpha_optimizer.step()


class SAC:
    def __init__(self, **kwargs):
        self.networks = ApproxContainer(**kwargs)
        self.gamma = kwargs['gamma']
        self.act_dist_cls = GaussDistribution

    def compute_gradient(self, data):
        start_time = time.time()

        obs = data['obs']
        logits = self.networks.policy(obs)
        act_dist = self.act_dist_cls(logits)
        new_act = act_dist.rsample()
        log_prob = act_dist.log_prob(new_act).sum(-1)
        data.update({
            'new_act': new_act,
            'log_prob': log_prob
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

        self.networks.alpha_optimizer.zero_grad()
        loss_alpha = self._compute_loss_alpha(data)
        loss_alpha.backward()

        value_grad = [p.grad.numpy() for p in self.networks.value.parameters()]
        q1_grad = [p.grad.numpy() for p in self.networks.q1.parameters()]
        q2_grad = [p.grad.numpy() for p in self.networks.q2.parameters()]
        policy_grad = [p.grad.numpy() for p in self.networks.policy.parameters()]
        alpha_grad = self.networks.log_alpha.grad.numpy()

        tb_info = {
            tb_tags['loss_critic']: loss_value.item(),
            tb_tags['loss_actor']: loss_policy.item(),
            tb_tags['critic_avg_value']: value.item(),
            'Train/critic_avg_q1': q1.item(),
            'Train/critic_avg_q2': q2.item(),
            'Train/entropy': entropy.item(),
            'Train/alpha': self.networks.log_alpha.exp().item(),
            tb_tags['alg_time']: (time.time() - start_time) * 1000
        }

        return value_grad + q1_grad + q2_grad + policy_grad + [alpha_grad], tb_info

    def _compute_loss_value(self, data):
        obs, new_act, log_prob = data['obs'], data['new_act'], data['log_prob']
        value = self.networks.value(obs).squeeze(-1)
        with torch.no_grad():
            q1 = self.networks.q1(obs, new_act)
            q2 = self.networks.q2(obs, new_act)
            target_value = torch.min(q1, q2) - self.networks.log_alpha.exp() * log_prob
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
        obs, new_act, log_prob = data['obs'], data['new_act'], data['log_prob']
        q1 = self.networks.q1(obs, new_act)
        q2 = self.networks.q2(obs, new_act)
        loss_policy = (self.networks.log_alpha.exp().detach() * log_prob -
                       torch.min(q1, q2)).mean()
        entropy = -log_prob.detach().mean()
        return loss_policy, entropy

    def _compute_loss_alpha(self, data):
        log_prob = data['log_prob']
        loss_alpha = -self.networks.log_alpha.exp() * \
                     (log_prob.detach() + self.networks.target_entropy).mean()
        return loss_alpha
