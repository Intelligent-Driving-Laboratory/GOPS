#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Distributed Soft Actor Critic Algorithm (DSAC)
#  Update: 2021-03-05, Ziqing Gu: create DSAC algorithm



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
        # create q networks
        q_func_type = kwargs['value_func_type']
        q_args = get_apprfunc_dict('q', q_func_type, **kwargs)
        self.q = create_apprfunc(**q_args)
        self.q_target = deepcopy(self.q)

        # create policy network
        policy_func_type = kwargs['policy_func_type']
        policy_args = get_apprfunc_dict('policy', policy_func_type, **kwargs)
        self.policy = create_apprfunc(**policy_args)
        self.policy_target =deepcopy(self.policy)

        # set target network gradients
        for p in self.policy_target.parameters():
            p.requires_grad = False
        for p in self.q_target.parameters():
            p.requires_grad = False

        # create optimizers
        self.q_optimizer = Adam(self.q.parameters(), lr=kwargs['q_learning_rate'])
        self.policy_optimizer = Adam(self.policy.parameters(), lr=kwargs['policy_learning_rate'])

    # create action_distributions
    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)

    def update(self, grads_info):
        q_grad = grads_info['q_grad']
        policy_grad = grads_info['policy_grad']
        polyak = 1 - grads_info['tau']
        iteration = grads_info['iteration']
        delay_update = grads_info['delay_update']

        # update q networks
        for p, grad in zip(self.q.parameters(), q_grad):
            p._grad = grad
        self.q_optimizer.step()

        # update policy network
        if iteration % delay_update == 0:
            for p, grad in zip(self.policy.parameters(), policy_grad):
                p._grad = grad
            self.policy_optimizer.step()
        # update target network
            with torch.no_grad():
                for p, p_targ in zip(self.q.parameters(), self.q_target.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1.0 - polyak) * p.data)

                for p, p_targ in zip(self.policy.parameters(), self.policy_target.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1.0 - polyak) * p.data)



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
        self.delay_update = kwargs['delay_update']

        if self.auto_alpha:
            self.log_alpha = torch.tensor(0, dtype=torch.float32)
            if self.use_gpu:
                self.log_alpha = self.log_alpha.cuda()
            self.log_alpha.requires_grad = True
            self.alpha_optimizer = Adam([self.log_alpha], lr=kwargs['alpha_learning_rate'])
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = kwargs['alpha']
            self.log_alpha = torch.log(torch.tensor(self.alpha))

        self.tb_info = {
            tb_tags['loss_actor']: 0,
            'Train/critic_avg_q': 0,
            'Train/entropy': 0,
            'Train/alpha': self.alpha,
            tb_tags['alg_time']: 0
        }

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
        params['delay_update'] = self.delay_update
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

        self.networks.q_optimizer.zero_grad()
        loss_q, q, std = self._compute_loss_q(data)
        loss_q.backward()
        self.tb_info['Train/critic_avg_q'] = q.item()
        self.tb_info['Train/critic_avg_std'] = std.item()
        if iteration % self.delay_update == 0:

            obs = data['obs']
            logits = self.networks.policy(obs)
            policy_mean = torch.tanh(logits[...,0]).mean().item()
            policy_std = logits[...,1].mean().item()

            act_dist = self.networks.create_action_distributions(logits)
            new_act, new_log_prob = act_dist.rsample()
            data.update({
                'new_act': new_act,
                'new_log_prob': new_log_prob
            })

            for p in self.networks.q.parameters():
                p.requires_grad = False

            self.networks.policy_optimizer.zero_grad()
            loss_policy, entropy = self._compute_loss_policy(data)
            loss_policy.backward()
            self.tb_info['Train/entropy'] = entropy.item()
            self.tb_info[tb_tags['loss_actor']] = loss_policy.item()
            self.tb_info['Train/policy_mean'] = policy_mean
            self.tb_info['Train/policy_std'] = policy_std
            for p in self.networks.q.parameters():
                p.requires_grad = True

            if self.auto_alpha:
                self.alpha_optimizer.zero_grad()
                loss_alpha = self._compute_loss_alpha(data)
                loss_alpha.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()

            self.tb_info['Train/alpha'] = self.alpha

        grad_info = {
            'q_grad': [p.grad for p in self.networks.q.parameters()],
            'policy_grad': [p.grad for p in self.networks.policy.parameters()],
            'tau': self.tau,
            'delay_update': self.delay_update,
            'iteration': iteration
        }


        self.tb_info[tb_tags['alg_time']] = (time.time() - start_time) * 1000

        return grad_info, self.tb_info

    def _q_evaluate(self, obs, act, qnet, min=False):
        StochaQ = qnet(obs, act)
        mean, log_std = StochaQ[..., 0], StochaQ[..., -1]
        std = log_std.exp()
        normal = Normal(torch.zeros(mean.shape), torch.ones(std.shape))
        if min == False:
            z = normal.sample()
            z = torch.clamp(z, -2, 2)
        elif min == True:
            z = -torch.abs(normal.sample())
        q_value = mean + torch.mul(z, std)  # + torch.mul(z, std)
        return mean, std, q_value

    def _compute_loss_q(self, data):
        obs, act, rew, obs2, done = \
            data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        logits_2 = self.networks.policy_target(obs2)
        act2_dist = self.networks.create_action_distributions(logits_2)
        act2, log_prob_act2 = act2_dist.rsample()

        q, q_std, q_sample = self._q_evaluate(obs, act, self.networks.q, min=False)
        # _, _, q_next_sample = self._q_evaluate(obs2, act2, self.networks.q_target, min=False)
        _, _, q_next_sample = self._q_evaluate(obs2, act2, self.networks.q_target, min=False)
        target_q, target_q_bound = self._compute_target_q(rew, done, q.detach(), q_std.detach(), q_next_sample.detach(),
                                                          log_prob_act2.detach())
        if self.bound:
            q_loss = torch.mean(torch.pow(q - target_q, 2) / (2 * torch.pow(q_std.detach(), 2)) \
                                + torch.pow(q.detach() - target_q_bound, 2) / (2 * torch.pow(q_std, 2)) \
                                + torch.log(q_std))
        else:
            q_loss = -Normal(q, q_std).log_prob(target_q).mean()
        return q_loss, q.detach().mean(), q_std.detach().mean()

    def _compute_target_q(self,r,done, q, q_std, q_next,log_prob_a_next):
        target_q = r + (1 - done) * self.gamma * (q_next - self.log_alpha.exp().detach() * log_prob_a_next)
        difference = torch.clamp(target_q - q, -self.TD_bound, self.TD_bound)
        target_q_bound = q + difference
        return target_q.detach(), target_q_bound.detach()

    def _compute_loss_policy(self, data):
        obs, new_act, new_log_prob = data['obs'], data['new_act'], data['new_log_prob']
        q, _, _ = self._q_evaluate(obs, new_act, self.networks.q, min=False)
        loss_policy = (self.alpha * new_log_prob - q).mean()
        entropy = -new_log_prob.detach().mean()
        return loss_policy, entropy

    def _compute_loss_alpha(self, data):
        new_log_prob = data['new_log_prob']
        loss_alpha = -self.log_alpha * (new_log_prob.detach() + self.target_entropy).mean()
        return loss_alpha


if __name__ == '__main__':
    print('this is dsac algorithm!')
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())