#   Copyright (c) 2020 ocp-tools Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Author: Sun Hao


__all__=['CartPoleDQN']


import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
import parl
import numpy as np


class CartPoleDQN(parl.Algorithm):
    def __init__(self, model, gamma, lr):
        """ 自定义的 DQN algorithm
        """
        super(CartPoleDQN,self).__init__(model)
        self.model = model
        self.target_model = copy.deepcopy(model)
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.apprfunc.to(device)
        #self.target_model.to(device)

        assert isinstance(gamma, float) # debug
        assert isinstance(lr, float)
        self.gamma = gamma
        self.lr = lr

        self.mse_loss = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def predict(self, obs_np):
        """
        """
        obs = torch.from_numpy(obs_np).float()
        with torch.no_grad():
            pred_q = self.model(obs)
        return pred_q

    def learn(self, obs_np, action_np, reward_np, next_obs_np, terminal_np):
        """ 类似监督学习
        进入数据为np.array
        从obs开始计算梯度
        """
        #-------------------------------
        obs = torch.from_numpy(obs_np).float()
        action = torch.from_numpy(action_np.astype(np.int64))
        reward = torch.from_numpy(reward_np).float()
        next_obs = torch.from_numpy(next_obs_np).float()
        terminal = torch.from_numpy(terminal_np).float()
        obs.requires_grad = True # 开始计算梯度
        # -------------------------------
        #print("action",action)
        temp = self.model(obs)
        #print(temp)
        pred_value = temp.gather(1, action)
        with torch.no_grad():
            max_v = self.target_model(next_obs).max(1, keepdim=True)[0]
            target = reward + (1 - terminal) * self.gamma * max_v
        self.optimizer.zero_grad()
        loss = self.mse_loss(pred_value, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        self.model.sync_weights_to(self.target_model)
