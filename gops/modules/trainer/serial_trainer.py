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
        ApproxContainer= getattr(file, 'ApproxContainer')
        self.networks = ApproxContainer(**kwargs)
        self.iteration = 0
        self.max_iteration = kwargs.get('max_iteration', 300)
        self.warm_size = kwargs['buffer_warm_size']
        self.batch_size = kwargs['batch_size']
        while self.buffer.size < self.warm_size:
            samples = self.sampler.sample()
            self.buffer.add_batch(samples)

        self.save_folder = kwargs['save_folder']
        self.log_save_interval = kwargs['log_save_interval']
        self.sample_sync_interval = kwargs['sample_sync_interval']
        self.apprfunc_save_interval = kwargs['apprfunc_save_interval']
        # setattr(self.alg, "writer", self.evaluator.writer)

    def step(self):
        # sampling
        if self.iteration % self.sample_sync_interval == 0:
            self.sampler.networks.load_state_dict(self.networks.state_dict())

        samples = self.sampler.sample()
        self.buffer.add_batch(samples)

        # replay
        samples = self.buffer.sample_batch(self.batch_size)

        # learning
        self.alg.networks.load_state_dict(self.networks.state_dict())
        grads = self.alg.compute_gradient(samples)

        # apply grad
        self.networks.update(grads, self.iteration)

        # evaluate
        if self.iteration % self.log_save_interval == 0:
            self.evaluator.networks.load_state_dict(self.networks.state_dict())
            self.evaluator.run_evaluation(self.iteration)

        # save
        if self.iteration % self.apprfunc_save_interval == 0:
            torch.save(self.networks.state_dict(), self.save_folder+'/apprfunc/apprfunc_{}.pkl'.format(self.iteration))

    def train(self):
        while self.iteration < self.max_iteration:
            # setattr(self.alg, "iteration", self.iteration)
            with Timer(self.evaluator.writer, step=self.iteration):
                self.step()
            if hasattr(self.alg, 'tb_info'):
                add_scalars(self.alg.tb_info, self.evaluator.writer, step=self.iteration)

            self.iteration += 1
            if self.iteration%10 == 0:
                print('Itertaion = ', self.iteration)

        self.evaluator.writer.flush()




