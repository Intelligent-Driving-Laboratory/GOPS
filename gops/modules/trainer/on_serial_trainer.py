#   Copyright (c) 2020 ocp-tools Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Author: Sun Hao
#  Update Date: 2021-01-03, Yuxuan JIANG & Guojian ZHAN : modified to allow discrete action space


__all__ = ['SerialTrainer']

import time
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from modules.utils.utils import Timer
from modules.utils.tensorboard_tools import add_scalars


class SerialTrainer():
    def __init__(self, alg, sampler, buffer, evaluator, **kwargs):
        self.alg = alg
        self.sampler = sampler
        self.buffer = buffer
        self.evaluator = evaluator

        ####import ddpg
        alg_name = kwargs['algorithm']
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        ApproxContainer = getattr(file, 'ApproxContainer')
        self.networks = ApproxContainer(**kwargs)
        self.iteration = 0
        self.max_iteration = kwargs.get('max_iteration', 300)
        self.warm_size = kwargs['buffer_warm_size']
        self.batch_size = kwargs['batch_size']
        self.ini_network_dir = kwargs['ini_network_dir']

        self.obsv_dim = kwargs['obsv_dim']
        self.act_dim = kwargs['action_dim']

        # initialize the networks
        if self.ini_network_dir is not None:
            self.networks.load_state_dict(torch.load(self.ini_network_dir))

        while self.buffer.size < self.warm_size:
            samples = self.sampler.sample()
            self.buffer.add_batch(samples)

        self.save_folder = kwargs['save_folder']
        self.log_save_interval = kwargs['log_save_interval']
        self.sample_sync_interval = kwargs['sample_sync_interval']
        self.apprfunc_save_interval = kwargs['apprfunc_save_interval']
        # setattr(self.alg, "writer", self.evaluator.writer)

    def step(self, iteration):
        with Timer(self.evaluator.writer, step=iteration):
            # sampling
            self.sampler.networks.load_state_dict(self.networks.state_dict())

            samples_with_replay_format = self.samples_conversion(self.sampler.sample())
            # learning
            self.alg.networks.load_state_dict(self.networks.state_dict())
            grads = self.alg.compute_gradient(samples_with_replay_format)

            # apply grad
            self.networks.update(grads, self.iteration)

        # evaluate
        if self.iteration % self.log_save_interval == 0:
            self.evaluator.networks.load_state_dict(self.networks.state_dict())
            self.evaluator.run_evaluation(self.iteration)

        # save
        if self.iteration % self.apprfunc_save_interval == 0:
            torch.save(self.networks.state_dict(),
                       self.save_folder + '/apprfunc/apprfunc_{}.pkl'.format(self.iteration))

    def train(self):
        while self.iteration < self.max_iteration:
            # setattr(self.alg, "iteration", self.iteration)

            self.step(self.iteration)

            if hasattr(self.alg, 'tb_info'):
                add_scalars(self.alg.tb_info, self.evaluator.writer, step=self.iteration)

            self.iteration += 1
            if self.iteration % 10 == 0:
                print('Itertaion = ', self.iteration)

        self.evaluator.writer.flush()

    def samples_conversion(self, samples):
        obs_tensor = torch.zeros(self.batch_size, self.obsv_dim)
        act_tensor = torch.zeros(self.batch_size, self.act_dim)
        obs2_tensor = torch.zeros(self.batch_size, self.obsv_dim)
        rew_tensor = torch.zeros(self.batch_size, )
        done_tensor = torch.zeros(self.batch_size, )
        idx = 0
        for sample in samples:
            obs, act, rew, next_obs, done = sample
            obs_tensor[idx] = torch.from_numpy(obs)
            act_tensor[idx] = torch.from_numpy(act)
            rew_tensor[idx] = torch.from_numpy(rew)
            obs2_tensor[idx] = torch.from_numpy(next_obs)
            done_tensor[idx] = torch.from_numpy(done)
        return dict(obs=obs_tensor, act=act_tensor, obs2=obs2_tensor, rew=rew_tensor,
                                          done=done_tensor)
