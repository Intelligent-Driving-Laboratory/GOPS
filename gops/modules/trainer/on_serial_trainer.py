#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yang GUAN
#  Description: Serial trainer for RL algorithms
#
#  Update Date: 2021-03-10, Wenhan CAO: Revise Codes
#  Update Date: 2021-05-21, Shengbo LI: Format Revise


__all__ = ['OnSerialTrainer']

import logging

import torch
from torch.utils.tensorboard import SummaryWriter

from modules.utils.tensorboard_tools import add_scalars

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from modules.utils.tensorboard_tools import tb_tags
import time


class OnSerialTrainer():
    def __init__(self, alg, sampler, evaluator, **kwargs):
        self.alg = alg

        self.sampler = sampler
        self.evaluator = evaluator

        # Import algorithm, appr func, sampler & buffer
        alg_name = kwargs['algorithm']
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        ApproxContainer = getattr(file, 'ApproxContainer')
        self.networks = ApproxContainer(**kwargs)
        self.iteration = 0
        self.max_iteration = kwargs.get('max_iteration')
        self.batch_size = kwargs['sample_batch_size']
        self.ini_network_dir = kwargs['ini_network_dir']
        self.obsv_dim = kwargs['obsv_dim']
        self.act_dim = kwargs['action_dim']

        # initialize the networks
        if self.ini_network_dir is not None:
            self.networks.load_state_dict(torch.load(self.ini_network_dir))

        self.save_folder = kwargs['save_folder']
        self.log_save_interval = kwargs['log_save_interval']
        self.apprfunc_save_interval = kwargs['apprfunc_save_interval']
        self.eval_interval = kwargs['eval_interval']
        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)
        self.writer.add_scalar(tb_tags['alg_time'], 0, 0)
        self.writer.add_scalar(tb_tags['sampler_time'], 0, 0)

        self.writer.flush()
        self.start_time = time.time()
        # setattr(self.alg, "writer", self.evaluator.writer)

    def step(self):
        # sampling
        self.sampler.networks.load_state_dict(self.networks.state_dict())
        samples, sampler_tb_dict = self.sampler.sample()
        samples_with_replay_format = self.samples_conversion(samples)

        # learning
        self.alg.networks.load_state_dict(self.networks.state_dict())
        grads, alg_tb_dict = self.alg.compute_gradient(samples_with_replay_format)

        # apply grad
        self.networks.update(grads, self.iteration)

        # log
        if self.iteration % self.log_save_interval == 0:
            print('Iter = ', self.iteration)
            add_scalars(alg_tb_dict, self.writer, step=self.iteration)
            add_scalars(sampler_tb_dict, self.writer, step=self.iteration)
        # evaluate
        if self.iteration % self.eval_interval == 0:
            self.evaluator.networks.load_state_dict(self.networks.state_dict())
            total_avg_return = self.evaluator.run_evaluation(self.iteration)
            self.writer.add_scalar(tb_tags['TAR of RL iteration'],
                                   total_avg_return,
                                   self.iteration)
            self.writer.add_scalar(tb_tags['TAR of total time'],
                                   total_avg_return,
                                   int(time.time() - self.start_time))
            self.writer.add_scalar(tb_tags['TAR of collected samples'],
                                   total_avg_return,
                                   self.sampler.get_total_sample_number())

        # save
        if self.iteration % self.apprfunc_save_interval == 0:
            torch.save(self.networks.state_dict(),
                       self.save_folder + '/apprfunc/apprfunc_{}.pkl'.format(self.iteration))

    def train(self):
        while self.iteration < self.max_iteration:
            self.step()
            self.iteration += 1

        self.writer.flush()

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
            rew_tensor[idx] = torch.tensor(rew)
            obs2_tensor[idx] = torch.from_numpy(next_obs)
            done_tensor[idx] = torch.tensor(done)
            idx += 1
        return dict(obs=obs_tensor, act=act_tensor, obs2=obs2_tensor, rew=rew_tensor,
                    done=done_tensor)
