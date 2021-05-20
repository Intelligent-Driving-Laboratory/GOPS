#   Copyright (c) 2020 ocp-tools Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#   Author: SUN-Hao
"""
class ApproxContainer

class DDPG
"""


__all__ = ['TD3']

import itertools
from copy import deepcopy
import torch
import torch.nn as nn
from torch.optim import Adam

from modules.create_pkg.create_apprfunc import create_apprfunc
from modules.utils.utils import get_apprfunc_dict


class ApproxContainer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.polyak = 1 - kwargs['tau']
        self.delay_update = kwargs['delay_update']
        # create value network
        q_args = get_apprfunc_dict('value', **kwargs)
        self.q1 = create_apprfunc(**q_args)
        self.q2 = create_apprfunc(**q_args)
        # create policy network
        policy_args = get_apprfunc_dict('policy', **kwargs)
        self.policy = create_apprfunc(**policy_args)
        # set network gradients
        for p in self.q1.parameters():
            p.requires_grad = True
        for p in self.q2.parameters():
            p.requires_grad = True
        for p in self.policy.parameters():
            p.requires_grad = True

        #  create target networks
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        self.policy_target = deepcopy(self.policy)
        # set target network gradients
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False
        for p in self.policy_target.parameters():
            p.requires_grad = False
        # set optimizers
        self.policy_optimizer = Adam(self.policy.parameters(), lr=kwargs['policy_learning_rate'])
        # self.q_params = itertools.chain(self.q1.parameters(), self.q2.parameters())
        self.q1_optimizer = Adam(self.q1.parameters(), lr=kwargs['value_learning_rate'])
        self.q2_optimizer = Adam(self.q2.parameters(), lr=kwargs['value_learning_rate'])

    def update(self, grads, iteration):
        # used by trainer to update networks
        q_grad_len = len(list(self.q1.parameters()))
        q1_grad, q2_grad,policy_grad = grads[:q_grad_len], grads[q_grad_len:2*q_grad_len],grads[2*q_grad_len:]

        # update q network
        for p, grad in zip(self.q1.parameters(), q1_grad):
            p._grad = torch.from_numpy(grad)
        for p, grad in zip(self.q2.parameters(), q2_grad):
            p._grad = torch.from_numpy(grad)

        self.q1_optimizer.step()
        self.q2_optimizer.step()
        # update policy network
        if iteration % self.delay_update == 0:
            for p, grad in zip(self.policy.parameters(), policy_grad):
                p._grad = torch.from_numpy(grad)
            self.policy_optimizer.step()
           # update target networks
            with torch.no_grad():
                for p, p_targ in zip(self.q1.parameters(), self.q1_target.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

                for p, p_targ in zip(self.q2.parameters(), self.q2_target.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

                for p, p_targ in zip(self.policy.parameters(), self.policy_target.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)


class TD3():
    def __init__(self, **kwargs):
        self.networks = ApproxContainer(**kwargs) # used in algorithm only for compute gradient of container
        self.gamma = kwargs['gamma']
        self.target_noise = kwargs.get('target_noise',0.2)
        self.noise_clip = kwargs.get('noise_clip',0.5)
        self.act_limit = kwargs['action_high_limit'][0]

    def compute_gradient(self, data):
        self.networks.q1_optimizer.zero_grad()
        self.networks.q2_optimizer.zero_grad()
        self.networks.policy_optimizer.zero_grad()

        loss_q = self._compute_loss_q(data)
        loss_q.backward()

        #----------------------------------
        for p in  self.networks.q1.parameters():
            p.requires_grad = False
        for p in self.networks.q2.parameters():
            p.requires_grad = False
        loss_policy = self._compute_loss_pi(data)
        loss_policy.backward()
        for p in  self.networks.q1.parameters():
            p.requires_grad = True
        for p in self.networks.q2.parameters():
            p.requires_grad = True
        # ----------------------------------

        q1_grad = [p._grad.numpy() for p in self.networks.q1.parameters()]
        q2_grad = [p._grad.numpy() for p in self.networks.q2.parameters()]
        policy_grad = [p._grad.numpy() for p in self.networks.policy.parameters()]
        return q1_grad +q2_grad + policy_grad

    def _compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q1 = self.networks.q1(o,a)
        q2 = self.networks.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = self.networks.policy_target(o2)
            # a2 = pi_targ
            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * self.target_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -self.act_limit, self.act_limit)

            # Target Q-values
            q1_pi_targ = self.networks.q1_target(o2, a2)
            q2_pi_targ = self.networks.q2_target(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    def _compute_loss_pi(self, data):
        o = data['obs']
        q1_pi = self.networks.q1(o, self.networks.policy(o))
        return -q1_pi.mean()


if __name__ == '__main__':
    print("Current algorithm is TD3, this script is not the entry of TD3 demo!")
    # a = True
    # b = 1 - a
    # print(b)
    # a = [1,2,3]
    # b = [4,5,6]
    # c = itertools.chain(a, b)
    # a[2] =222
    # print(list(c))

