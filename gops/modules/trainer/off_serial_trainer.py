#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yang GUAN
#  Description: Serial trainer for RL algorithms
#
#  Update Date: 2021-03-10, Wenhan CAO: Revise Codes
#  Update Date: 2021-05-21, Shengbo LI: Format Revise


__all__ = ['OffSerialTrainer']

import torch
import logging
from modules.utils.utils import Timer
from modules.utils.tensorboard_tools import add_scalars
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from modules.utils.tensorboard_tools import tb_tags


class OffSerialTrainer():
    def __init__(self, alg, sampler, buffer, evaluator, **kwargs):
        self.alg = alg

        self.sampler = sampler
        self.buffer = buffer
        self.evaluator = evaluator

        # Import algorithm, appr func, sampler & buffer
        alg_name = kwargs['algorithm']
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        ApproxContainer = getattr(file, 'ApproxContainer')
        self.networks = ApproxContainer(**kwargs)
        self.iteration = 0
        self.max_iteration = kwargs.get('max_iteration')
        self.warm_size = kwargs['buffer_warm_size']
        self.replay_batch_size = kwargs['replay_batch_size']

        # Collect enough warm samples
        while self.buffer.size < self.warm_size:
            samples = self.sampler.sample()
            self.buffer.add_batch(samples)

        self.save_folder = kwargs['save_folder']
        self.log_save_interval = kwargs['log_save_interval']
        self.sampler_sync_interval = kwargs['sampler_sync_interval']
        self.apprfunc_save_interval = kwargs['apprfunc_save_interval']
        self.eval_interval = kwargs['eval_interval']
        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)
        self.writer.add_scalar(tb_tags['time'], 0, 0)

        self.writer.flush()
        # setattr(self.alg, "writer", self.evaluator.writer)

    def step(self):
        # sampling
        if self.iteration % self.sampler_sync_interval == 0:
            self.sampler.networks.load_state_dict(self.networks.state_dict())

        samples = self.sampler.sample()
        self.buffer.add_batch(samples)

        # replay
        samples = self.buffer.sample_batch(self.replay_batch_size)

        # learning
        self.alg.networks.load_state_dict(self.networks.state_dict())
        grads, alg_tb_dict = self.alg.compute_gradient(samples)

        # apply grad
        self.networks.update(grads, self.iteration)

        # log
        if self.iteration % self.log_save_interval == 0:
            print('Iter = ', self.iteration)
            add_scalars(alg_tb_dict, self.writer, step=self.iteration)

        # evaluate
        if self.iteration % self.eval_interval == 0:
            self.evaluator.networks.load_state_dict(self.networks.state_dict())
            self.writer.add_scalar(tb_tags['total_average_return'], self.evaluator.run_evaluation(),
                                   self.iteration)

        # save
        if self.iteration % self.apprfunc_save_interval == 0:
            torch.save(self.networks.state_dict(),
                       self.save_folder + '/apprfunc/apprfunc_{}.pkl'.format(self.iteration))

    def train(self):
        while self.iteration < self.max_iteration:
            self.step()
            self.iteration += 1

        self.writer.flush()
