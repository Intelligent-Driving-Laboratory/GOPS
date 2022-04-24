#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Serial trainer for RL algorithms
#  Update Date: 2021-05-21, Shengbo LI: Format Revise
#  Update Date: 2022-04-14, Jiaxin Gao: decrease parameters copy times

__all__ = ["OffSerialTrainer"]

import logging
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from gops.utils.tensorboard_tools import add_scalars
from gops.utils.utils import ModuleOnDevice

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from gops.utils.tensorboard_tools import tb_tags


class OffSerialTrainer:
    def __init__(self, alg, sampler, buffer, evaluator, **kwargs):
        self.alg = alg
        self.sampler = sampler
        self.buffer = buffer
        self.evaluator = evaluator

        self.networks = self.alg.networks
        self.sampler.networks = self.networks
        self.evaluator.networks = self.networks

        self.iteration = 0
        self.max_iteration = kwargs.get("max_iteration")
        self.warm_size = kwargs["buffer_warm_size"]
        self.replay_batch_size = kwargs["replay_batch_size"]
        self.ini_network_dir = kwargs["ini_network_dir"]

        # initialize the networks
        if self.ini_network_dir is not None:
            self.networks.load_state_dict(torch.load(self.ini_network_dir))

        # Collect enough warm samples
        while self.buffer.size < self.warm_size:
            samples, sampler_tb_dict = self.sampler.sample()
            self.buffer.add_batch(samples)

        self.save_folder = kwargs["save_folder"]
        self.log_save_interval = kwargs["log_save_interval"]
        self.apprfunc_save_interval = kwargs["apprfunc_save_interval"]
        self.eval_interval = kwargs["eval_interval"]
        self.sample_interval = kwargs.get("sample_interval", 1)
        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)
        self.writer.add_scalar(tb_tags["alg_time"], 0, 0)
        self.writer.add_scalar(tb_tags["sampler_time"], 0, 0)
        self.start_time = time.time()
        self.writer.flush()
        # setattr(self.alg, "writer", self.evaluator.writer)

        self.use_gpu = kwargs["use_gpu"]
        if self.use_gpu:
            self.networks.cuda()

    def step(self):
        # sampling
        sampler_tb_dict = {}
        if self.iteration % self.sample_interval == 0:
            with ModuleOnDevice(self.networks, 'cpu'):
                sampler_samples, sampler_tb_dict = self.sampler.sample()
            self.buffer.add_batch(sampler_samples)

        # replay
        replay_samples = self.buffer.sample_batch(self.replay_batch_size)

        # learning
        if self.use_gpu:
            for k, v in replay_samples.items():
                replay_samples[k] = v.cuda()
        alg_tb_dict = self.alg.local_update(replay_samples, self.iteration)

        # log
        if self.iteration % self.log_save_interval == 0:
            print("Iter = ", self.iteration)
            add_scalars(alg_tb_dict, self.writer, step=self.iteration)
            add_scalars(sampler_tb_dict, self.writer, step=self.iteration)
        # evaluate
        if self.iteration % self.eval_interval == 0:
            with ModuleOnDevice(self.networks, 'cpu'):
                total_avg_return = self.evaluator.run_evaluation(self.iteration)
            self.writer.add_scalar(
                tb_tags["Buffer RAM of RL iteration"],
                self.buffer.__get_RAM__(),
                self.iteration,
            )
            self.writer.add_scalar(
                tb_tags["TAR of RL iteration"], total_avg_return, self.iteration
            )
            self.writer.add_scalar(
                tb_tags["TAR of replay samples"],
                total_avg_return,
                self.iteration * self.replay_batch_size,
            )
            self.writer.add_scalar(
                tb_tags["TAR of total time"],
                total_avg_return,
                int(time.time() - self.start_time),
            )
            self.writer.add_scalar(
                tb_tags["TAR of collected samples"],
                total_avg_return,
                self.sampler.get_total_sample_number(),
            )

        # save
        if self.iteration % self.apprfunc_save_interval == 0:
            torch.save(
                self.networks.state_dict(),
                self.save_folder + "/apprfunc/apprfunc_{}.pkl".format(self.iteration),
            )

    def train(self):
        while self.iteration < self.max_iteration:
            self.step()
            self.iteration += 1

        self.writer.flush()
