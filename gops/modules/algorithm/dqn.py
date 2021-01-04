#   Copyright (c) 2020 ocp-tools Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Author: Sun Hao
#  Update Date: 2021-01-03, Yuxuan JIANG & Guojian ZHAN : implement DQN


__all__=['DQN']


from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from modules.create_pkg.create_apprfunc import create_apprfunc

class DQN():
    def __init__(self, learning_rate=0.001, gamma=0.995, tau=0.005,**kwargs):
        """Deep Q-Network (DQN) algorithm

        A DQN implementation with soft target update.

        Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control through deep reinforcement learning.
        Nature 518, 529–533 (2015). https://doi.org/10.1038/nature14236
        
        Args:
            learning_rate (float, optional): Q network learning rate. Defaults to 0.001.
            gamma (float, optional): Discount factor. Defaults to 0.995.
            tau (float, optional): Average factor. Defaults to 0.005.
        """
        Q_network_dict = {'apprfunc':kwargs['apprfunc'],
                       'name':kwargs['value_func_name'],
                       'obs_dim': kwargs['obsv_dim'],
                       'act_dim': kwargs['action_num'],
                       'hidden_sizes': kwargs['value_hidden_sizes'],
                       'activation': kwargs['value_output_activation'],
                       }
        Q_network = create_apprfunc(**Q_network_dict)
        target_network = deepcopy(Q_network)

        self.apprfunc = Q_network
        self.target_apprfunc =  target_network
        self.target_apprfunc.eval()

        self.polyak = 1 - tau
        self.gamma = gamma
        self.lr = learning_rate
        self.optimizer = Adam(self.apprfunc.parameters(), lr=self.lr)

    def learn(self, data):
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()

        # Soft update
        with torch.no_grad():
            for p, p_targ in zip(self.apprfunc.parameters(), self.target_apprfunc.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def predict(self,obs):
        with torch.no_grad():
            return self.apprfunc(obs).argmax().item()

    def compute_loss(self,data):  
        obs, action, reward, next_obs, done = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q_policy = self.apprfunc(obs).gather(1, action.to(torch.long)).squeeze()

        with torch.no_grad():
            q_target,_ = torch.max(self.target_apprfunc(next_obs),1)
        q_expect = reward +  self.gamma * (1 - done) * q_target

        loss = F.mse_loss(q_policy, q_expect)

        return loss
