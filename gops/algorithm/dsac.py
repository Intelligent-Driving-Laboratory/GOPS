#   Copyright (c) 2020 ocp-tools Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Author: Gu ziqing


__all__ = ['ApproxContainer','DSAC']

from copy import deepcopy
import torch
import torch.nn as nn
from torch.optim import Adam
import warnings
import time
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.utils import get_apprfunc_dict
from gops.utils.action_distributions import GaussDistribution
from gops.utils.tensorboard_tools import tb_tags
from torch.distributions import Normal

class ApproxContainer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
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
        self.q1_optimizer = Adam(self.q1.parameters(), lr=kwargs['q1_learning_rate'])
        self.q2_optimizer = Adam(self.q2.parameters(), lr=kwargs['q2_learning_rate'])
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



class DSAC:
    def __init__(self, **kwargs):
        self.networks = ApproxContainer(**kwargs)
        self.use_gpu = kwargs['enable_cuda']
        self.gamma = kwargs['gamma']
        self.tau = kwargs['tau']
        self.reward_scale = kwargs['reward_scale']
        self.target_entropy = -kwargs['action_dim']
        self.auto_alpha = kwargs['auto_alpha']
        self.TD_bound = kwargs['TD_bound']
        self.bound = kwargs['bound']
        if self.auto_alpha:
            self.log_alpha = torch.tensor(0, dtype=torch.float32)
            if self.use_gpu:
                self.log_alpha = self.log_alpha.cuda()
            self.log_alpha.requires_grad = True
            self.alpha_optimizer = Adam([self.log_alpha], lr=kwargs['alpha_learning_rate'])
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = kwargs['alpha']

    def set_parameters(self, param_dict):
        for key in param_dict:
            if hasattr(self, key):
                setattr(self, key, param_dict[key])
            else:
                warning_msg = "param '" + key + "' is not defined in algorithm!"
                warnings.warn(warning_msg)

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)

    def get_parameters(self):
        params = dict()
        params['gamma'] = self.gamma
        params['tau'] = self.tau
        params['use_gpu'] = self.use_gpu
        params['auto_alpha'] = self.auto_alpha
        params['alpha'] = self.alpha
        params['reward_scale'] = self.reward_scale
        params['target_entropy'] = self.target_entropy
        params['TD_bound'] = self.TD_bound
        params['bound'] = self.bound
        return params

    def compute_gradient(self, data:dict, iteration):
        start_time = time.time()
        data['rew'] = data['rew']*self.reward_scale
        if self.use_gpu:
            self.networks = self.networks.cuda()
            for k, v in data.items():
                data[k] = v.cuda()

        obs = data['obs']
        logits = self.networks.policy(obs)
        act_dist = self.networks.create_action_distributions(logits)
        new_act,new_log_prob = act_dist.rsample()
        data.update({
            'new_act': new_act,
            'new_log_prob': new_log_prob
        })

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

        grad_info = {
            'value_grad': [p.grad for p in self.networks.value.parameters()],
            'q1_grad': [p.grad for p in self.networks.q1.parameters()],
            'q2_grad': [p.grad for p in self.networks.q2.parameters()],
            'policy_grad': [p.grad for p in self.networks.policy.parameters()],
            'tau': self.tau
        }

        tb_info = {
            tb_tags['loss_actor']: loss_policy.item(),
            'Train/critic_avg_q1': q1.item(),
            'Train/critic_avg_q2': q2.item(),
            'Train/entropy': entropy.item(),
            'Train/alpha': self.alpha,
            tb_tags['alg_time']: (time.time() - start_time) * 1000
        }

        return grad_info, tb_info

    def _q_evaluate(self, obs, act, qnet, min=False):
        StochaQ = qnet(obs, act)
        mean, log_std = StochaQ[..., 0].unsqueeze(1), StochaQ[..., -1].unsqueeze(1)
        std = log_std.exp()
        normal = Normal(torch.zeros(mean.shape), torch.ones(std.shape))
        if min == False:
            z = normal.sample()
            z = torch.clamp(z, -2, 2)
        elif min == True:
            z = -torch.abs(normal.sample())
        q_value_tmp = mean + torch.mul(z, std)
        q_value = torch.squeeze(q_value_tmp, -1)  # + torch.mul(z, std)
        return mean, std, q_value

    def _compute_loss_value(self, data):
        obs, new_act, new_log_prob = data['obs'], data['new_act'], data['new_log_prob']
        value = self.networks.value(obs).squeeze(-1)
        with torch.no_grad():
            _, _, q1 = self._q_evaluate(obs, new_act, self.networks.q1, min=False)
            _, _, q2 = self._q_evaluate(obs, new_act, self.networks.q2, min=False)
            target_value = torch.min(q1, q2) - self.alpha * new_log_prob
        loss_value = ((value - target_value) ** 2).mean()
        return loss_value, value.detach().mean()

    def _compute_loss_q(self, data):
        obs, act, rew, obs2, done = \
            data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        act2, new_log_prob = data['new_act'], data['new_log_prob']
        # _, _, q1 = self._q_evaluate(obs, act, self.networks.q1, min=False)
        # _, _, q2 = self._q_evaluate(obs, act, self.networks.q2, min=False)
        # with torch.no_grad():
        #     next_value = self.networks.value_target(obs2).squeeze(-1)
        #     backup = rew + (1 - done) * self.gamma * next_value
        # loss_q1 = ((q1 - backup) ** 2).mean()
        # loss_q2 = ((q2 - backup) ** 2).mean()
        # return loss_q1 + loss_q2, q1.detach().mean(), q2.detach().mean()
        _, q_std1, q1 = self._q_evaluate(obs, act, self.networks.q1, min=False)
        _, q_std2, q2 = self._q_evaluate(obs, act, self.networks.q2, min=False)
        _, _, q_next_sample1 = self._q_evaluate(obs2, act2, self.networks.q1, min=False)
        _, _, q_next_sample2 = self._q_evaluate(obs2, act2, self.networks.q2, min=False)

        q = torch.min(q1, q2)
        q_std = torch.min(q_std1, q_std2)
        q_next_target = torch.min(q_next_sample1, q_next_sample2)
        # a_next, log_prob_a_next, _ = self._policy_evaluate(data, epsilon=1e-4)
        target_q, target_q_bound = self._compute_target_q(rew, done, q.detach(), q_std.detach(), q_next_target.detach(),
                                                          new_log_prob.detach())
        if self.bound:
            q_loss = torch.mean(torch.pow(q - target_q, 2) / (2 * torch.pow(q_std.detach(), 2)) \
                                + torch.pow(q.detach() - target_q_bound, 2) / (2 * torch.pow(q_std, 2)) \
                                + torch.log(q_std))
        else:
            q_loss = -Normal(q, q_std).log_prob(target_q).mean()
        return q_loss, q1.detach().mean(), q2.detach().mean()

    def _compute_target_q(self,r,done, q, q_std, q_next,log_prob_a_next):
        target_q = r + (1 - done) * self.gamma * (q_next - self.log_alpha.exp().detach() * log_prob_a_next)
        # target_max = q + 10 * q_std
        # target_min = q - 10 * q_std
        # target_q = torch.min(target_q, target_max)
        # target_q = torch.max(target_q, target_min)
        difference = torch.clamp(target_q - q, -self.TD_bound, self.TD_bound)
        target_q_bound = q + difference
        return target_q.detach(), target_q_bound.detach()

    def _compute_loss_policy(self, data):
        obs, new_act, new_log_prob = data['obs'], data['new_act'], data['new_log_prob']
        _, _, q1 = self._q_evaluate(obs, new_act, self.networks.q1, min=False)
        _, _, q2 = self._q_evaluate(obs, new_act, self.networks.q2, min=False)
        loss_policy = (self.alpha * new_log_prob - torch.min(q1, q2)).mean()
        entropy = -new_log_prob.detach().mean()
        return loss_policy, entropy

    def _compute_loss_alpha(self, data):
        new_log_prob = data['new_log_prob']
        loss_alpha = -self.log_alpha.exp() * (new_log_prob.detach() + self.target_entropy).mean()
        return loss_alpha


if __name__ == '__main__':
    print('this is dsac algorithm!')
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())