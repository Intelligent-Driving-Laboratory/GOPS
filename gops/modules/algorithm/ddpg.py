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
#  Comments: ?


__all__ = ['DDPG']

from copy import deepcopy
import torch
import torch.nn as nn
from torch.optim import Adam

from modules.create_pkg.create_apprfunc import create_apprfunc
from modules.utils.utils import get_apprfunc_dict
from modules.utils.utils import ActorCriticApprFunc


class DDPG():
    def __init__(self,**kwargs):

        critic_q_args = get_apprfunc_dict('value',**kwargs)
        critic = create_apprfunc(**critic_q_args)

        actor_args = get_apprfunc_dict('policy',**kwargs)
        actor = create_apprfunc(**actor_args)

        self.apprfunc = ActorCriticApprFunc(actor,critic)
        self.target_apprfunc =  deepcopy(self.apprfunc)

        for p in self.target_apprfunc.parameters():
            p.requires_grad = False

        self.gamma = kwargs['gamma']
        self.polyak = 1 - kwargs['tau']
        self.pi_optimizer = Adam(self.apprfunc.pi.parameters(), lr=kwargs['policy_learning_rate']) #
        self.q_optimizer = Adam(self.apprfunc.q.parameters(), lr=kwargs['value_learning_rate'])


    def learn(self,data):
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        for p in self.apprfunc.q.parameters():
            p.requires_grad = False

        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        for p in self.apprfunc.q.parameters():
            p.requires_grad = True

        with torch.no_grad():
            for p, p_targ in zip(self.apprfunc.parameters(), self.target_apprfunc.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)


    def predict(self,obs):
        return self.apprfunc.act(obs)

    def compute_loss_q(self,data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q = self.apprfunc.q(o,a)

        with torch.no_grad():
            q_pi_targ = self.target_apprfunc.q(o2, self.target_apprfunc.pi(o2))
            backup = r +  self.gamma * (1 - d) * q_pi_targ

        loss_q = ((q - backup)**2).mean()
        return loss_q

    def compute_loss_pi(self,data):
        o = data['obs']
        q_pi = self.apprfunc.q(o, self.apprfunc.pi(o))
        return -q_pi.mean()


if __name__ == '__main__':
    pass
