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


__all__ = ['DDPG']

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
        self.q = create_apprfunc(**q_args)
        # create policy network
        policy_args = get_apprfunc_dict('policy', **kwargs)
        self.policy = create_apprfunc(**policy_args)
        #  create target networks
        self.q_target = deepcopy(self.q)
        self.policy_target = deepcopy(self.policy)
        # set target network gradients
        for p in self.q_target.parameters():
            p.requires_grad = False
        for p in self.policy_target.parameters():
            p.requires_grad = False
        # set optimizers
        self.policy_optimizer = Adam(self.policy.parameters(), lr=kwargs['policy_learning_rate'])
        self.q_optimizer = Adam(self.q.parameters(), lr=kwargs['value_learning_rate'])

    def update(self, grads, iteration):
        q_grad_len = len(list(self.q.parameters()))
        q_grad, policy_grad = grads[:q_grad_len], grads[q_grad_len:]
        #  zip()  : [()], list[tuple]
        for p, grad in zip(self.q.parameters(), q_grad):
            p._grad = torch.from_numpy(grad)
        for p, grad in zip(self.policy.parameters(), policy_grad):
            p._grad = torch.from_numpy(grad)
        # update q network
        self.q_optimizer.step()
        # update policy network
        if iteration % self.delay_update == 0:
            self.policy_optimizer.step()
       # update target networks
        with torch.no_grad():
            for p, p_targ in zip(self.q.parameters(), self.q_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(self.policy.parameters(), self.policy_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)


class DDPG():
    def __init__(self, **kwargs):
        self.networks = ApproxContainer(**kwargs)
        self.gamma = kwargs['gamma']
        self.polyak = 1 - kwargs['tau']
        self.policy_optimizer = Adam(self.networks.policy.parameters(), lr=kwargs['policy_learning_rate'])  #
        self.q_optimizer = Adam(self.networks.q.parameters(), lr=kwargs['value_learning_rate'])

    def compute_gradient(self, data):
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()

        for p in self.networks.q.parameters():
            p.requires_grad = False

        self.policy_optimizer.zero_grad()
        loss_policy = self.compute_loss_policy(data)
        loss_policy.backward()

        for p in self.networks.q.parameters():
            p.requires_grad = True

        q_grad = [p._grad.numpy() for p in self.networks.q.parameters()]
        policy_grad = [p._grad.numpy() for p in self.networks.policy.parameters()]
        return q_grad + policy_grad

    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q = self.networks.q(o, a)

        with torch.no_grad():
            q_policy_targ = self.networks.q_target(o2, self.networks.policy(o2))
            backup = r + self.gamma * (1 - d) * q_policy_targ

        loss_q = ((q - backup) ** 2).mean()
        return loss_q

    def compute_loss_policy(self, data):
        o = data['obs']
        q_policy = self.networks.q(o, self.networks.policy(o))
        return -q_policy.mean()


if __name__ == '__main__':
    pass
