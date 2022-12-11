#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Serial trainer for on-policy RL algorithms
#  Update Date: 2021-03-10, Wenhan CAO: Revise Codes
#  Update Date: 2021-05-21, Shengbo LI: Format Revise
#  Update: 2022-12-05, Wenhan Cao: add annotation

__all__ = ["OnSerialTrainer"]

from cmath import inf
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from gops.utils.tensorboard_setup import add_scalars
from gops.utils.tensorboard_setup import tb_tags
from gops.utils.common_utils import ModuleOnDevice


class OnSerialTrainer:
    def __init__(self, alg, sampler, evaluator, **kwargs):
        self.alg = alg
        self.sampler = sampler
        self.evaluator = evaluator

        # create center network
        self.networks = self.alg.networks
        self.sampler.networks = self.networks
        self.evaluator.networks = self.networks

        # initialize center network
        if kwargs["ini_network_dir"] is not None:
            self.networks.load_state_dict(torch.load(kwargs["ini_network_dir"]))

        self.max_iteration = kwargs.get("max_iteration")
        self.log_save_interval = kwargs["log_save_interval"]
        self.apprfunc_save_interval = kwargs["apprfunc_save_interval"]
        self.eval_interval = kwargs["eval_interval"]
        self.best_tar = -inf
        self.save_folder = kwargs["save_folder"]
        self.iteration = 0

        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)
        # flush tensorboard at the beginning
        add_scalars(
            {tb_tags["alg_time"]: 0, tb_tags["sampler_time"]: 0}, self.writer, 0
        )
        self.writer.flush()

        self.use_gpu = kwargs["use_gpu"]

        self.start_time = time.time()

    def step(self):
        # sampling
        (
            samples_with_replay_format,
            sampler_tb_dict,
        ) = self.sampler.sample_with_replay_format()

        # learning
        if self.use_gpu:
            for k, v in samples_with_replay_format.items():
                samples_with_replay_format[k] = v.cuda()
        with ModuleOnDevice(self.networks, "cuda" if self.use_gpu else "cpu"):
            alg_tb_dict = self.alg.local_update(
                samples_with_replay_format, self.iteration
            )

        # log
        if self.iteration % self.log_save_interval == 0:
            print("Iter = ", self.iteration)
            add_scalars(alg_tb_dict, self.writer, step=self.iteration)
            add_scalars(sampler_tb_dict, self.writer, step=self.iteration)

        # evaluate
        if self.iteration % self.eval_interval == 0:
            total_avg_return = self.evaluator.run_evaluation(self.iteration)

            if (
                total_avg_return >= self.best_tar
                and self.iteration >= self.max_iteration / 5
            ):
                self.best_tar = total_avg_return
                print("Best return = {}!".format(str(self.best_tar)))

                for filename in os.listdir(self.save_folder + "/apprfunc/"):
                    if filename.endswith("_opt.pkl"):
                        os.remove(self.save_folder + "/apprfunc/" + filename)

                torch.save(
                    self.networks.state_dict(),
                    self.save_folder
                    + "/apprfunc/apprfunc_{}_opt.pkl".format(self.iteration),
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
                self.sampler.get_total_sample_number(),
            )

        # save
        if self.iteration % self.apprfunc_save_interval == 0:
            self.save_apprfunc()

    def train(self):
        while self.iteration < self.max_iteration:
            self.step()
            self.iteration += 1

        self.save_apprfunc()
        self.writer.flush()

    def save_apprfunc(self):
        torch.save(
            self.networks.state_dict(),
            self.save_folder + "/apprfunc/apprfunc_{}.pkl".format(self.iteration),
        )
