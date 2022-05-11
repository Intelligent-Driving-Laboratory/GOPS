#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Serial trainer for RL algorithms
#  Update Date: 2021-03-10, Wenhan CAO: Revise Codes


__all__ = ["OnSyncTrainer"]

import logging
import random
import time

import ray
import torch
from torch.utils.tensorboard import SummaryWriter

from gops.utils.tensorboard_tools import add_scalars

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from gops.utils.tensorboard_tools import tb_tags
import warnings
import importlib

warnings.filterwarnings("ignore")


class OnSyncTrainer:
    def __init__(self, alg, sampler, evaluator, **kwargs):
        self.alg = alg
        self.samplers = sampler
        self.evaluator = evaluator
        self.iteration = 0
        self.max_iteration = kwargs["max_iteration"]
        self.ini_network_dir = kwargs["ini_network_dir"]
        self.save_folder = kwargs["save_folder"]
        self.log_save_interval = kwargs["log_save_interval"]
        self.apprfunc_save_interval = kwargs["apprfunc_save_interval"]
        self.iteration = 0
        self.save_folder = kwargs["save_folder"]
        self.log_save_interval = kwargs["log_save_interval"]
        self.apprfunc_save_interval = kwargs["apprfunc_save_interval"]
        self.eval_interval = kwargs["eval_interval"]
        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)
        self.writer.add_scalar(tb_tags["alg_time"], 0, 0)
        self.writer.add_scalar(tb_tags["sampler_time"], 0, 0)

        self.writer.flush()

        # create center network
        alg_name = kwargs["algorithm"]
        alg_file_name = alg_name.lower()
        try:
            module = importlib.import_module("gops.algorithm." + alg_file_name)
        except NotImplementedError:
            raise NotImplementedError("This algorithm does not exist")
        if hasattr(module, alg_name):
            alg_cls = getattr(module, alg_name)
            alg_net = alg_cls(**kwargs)
        else:
            raise NotImplementedError("This algorithm is not properly defined")

        self.networks = alg_net

        self.ini_network_dir = kwargs["ini_network_dir"]

        # initialize the networks
        if self.ini_network_dir is not None:
            self.networks.load_state_dict(torch.load(self.ini_network_dir))

        self.use_gpu = kwargs["use_gpu"]
        if self.use_gpu:
            self.alg.networks.cuda()

        self.start_time = time.time()

    def step(self):
        # sampling
        weights = ray.put(self.networks.state_dict())  # 把中心网络的参数放在底层内存里面
        for sampler in self.samplers:  # 对每个完成的sampler，
            sampler.load_state_dict.remote(weights)  # 同步sampler的参数
        samples, sampler_tb_dict = zip(
            *ray.get(
                [
                    sampler.sample_with_replay_format.remote()
                    for sampler in self.samplers
                ]
            )
        )
        sampler_tb_dict = sampler_tb_dict[0]
        all_samples = concate(samples)

        # learning
        if self.use_gpu:
            for k, v in all_samples.items():
                all_samples[k] = v.cuda()
        alg_tb_dict = self.alg.local_update(all_samples, self.iteration)  # 更新learner参数
        self.networks.load_state_dict(self.alg.state_dict())

        # log
        if self.iteration % self.log_save_interval == 0:
            print("Iter = ", self.iteration)
            add_scalars(alg_tb_dict, self.writer, step=self.iteration)
            add_scalars(sampler_tb_dict, self.writer, step=self.iteration)

        # evaluate
        if self.iteration % self.eval_interval == 0:
            # calculate total sample number
            self.evaluator.load_state_dict.remote(self.networks.state_dict())
            total_avg_return = ray.get(
                self.evaluator.run_evaluation.remote(self.iteration)
            )
            self.writer.add_scalar(
                tb_tags["TAR of RL iteration"], total_avg_return, self.iteration
            )
            self.writer.add_scalar(
                tb_tags["TAR of total time"],
                total_avg_return,
                int(time.time() - self.start_time),
            )
            self.writer.add_scalar(
                tb_tags["TAR of collected samples"],
                total_avg_return,
                sum(
                    ray.get(
                        [
                            sampler.get_total_sample_number.remote()
                            for sampler in self.samplers
                        ]
                    )
                ),
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


def concate(samples):
    all_samples = {}
    for key in samples[0].keys():
        if samples[0][key] is not None:
            all_samples[key] = torch.cat([sample[key] for sample in samples], dim=0)
    return all_samples
