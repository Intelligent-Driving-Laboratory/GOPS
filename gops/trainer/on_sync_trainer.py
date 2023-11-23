#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Synchronous Parallel Trainer for on-policy RL algorithms
#  Update Date: 2021-03-10, Wenhan CAO: Revise Codes
#  Update: 2022-12-05, Wenhan Cao: add annotation

__all__ = ["OnSyncTrainer"]

from cmath import inf
import importlib
import os
import time
import warnings

import ray
import torch
from torch.utils.tensorboard import SummaryWriter

from gops.utils.parallel_task_manager import TaskPool
from gops.utils.tensorboard_setup import add_scalars, tb_tags
from gops.utils.log_data import LogData

warnings.filterwarnings("ignore")


class OnSyncTrainer:
    def __init__(self, alg, sampler, evaluator, **kwargs):
        self.alg = alg
        self.samplers = sampler
        self.evaluator = evaluator

        # create center network
        alg_name = kwargs["algorithm"]
        alg_file_name = alg_name.lower()
        try:
            module = importlib.import_module("gops.algorithm." + alg_file_name)
        except NotImplementedError:
            raise NotImplementedError("This algorithm does not exist")
        if hasattr(module, alg_name):
            alg_cls = getattr(module, alg_name)
            self.networks = alg_cls(**kwargs)
        else:
            raise NotImplementedError("This algorithm is not properly defined")

        # initialize center network
        if kwargs["ini_network_dir"] is not None:
            self.networks.load_state_dict(torch.load(kwargs["ini_network_dir"]))

        self.max_iteration = kwargs["max_iteration"]
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

        self.sampler_tb_dict = LogData()

        # create evaluation tasks
        self.evluate_tasks = TaskPool()
        self.last_eval_iteration = 0

        self.use_gpu = kwargs["use_gpu"]
        if self.use_gpu:
            self.alg.networks.cuda()
        self.alg.networks.train()

        self.start_time = time.time()

    def step(self):
        # sampling
        weights = ray.put(self.networks.state_dict())
        for sampler in self.samplers:
            sampler.load_state_dict.remote(weights)
        samples, sampler_tb_dict = zip(
            *ray.get(
                [
                    sampler.sample_with_replay_format.remote()
                    for sampler in self.samplers
                ]
            )
        )
        self.sampler_tb_dict.add_average(sampler_tb_dict)
        all_samples = concate(samples)

        # learning
        if self.use_gpu:
            for k, v in all_samples.items():
                all_samples[k] = v.cuda()
        alg_tb_dict = self.alg.local_update(all_samples, self.iteration)
        self.networks.load_state_dict(self.alg.state_dict())

        # log
        if self.iteration % self.log_save_interval == 0:
            print("Iter = ", self.iteration)
            add_scalars(alg_tb_dict, self.writer, step=self.iteration)
            add_scalars(self.sampler_tb_dict.pop(), self.writer, step=self.iteration)

        # save
        if self.iteration % self.apprfunc_save_interval == 0:
            self.save_apprfunc()

        # evaluate
        if self.iteration - self.last_eval_iteration >= self.eval_interval:
            if self.evluate_tasks.count == 0:
                # There is no evaluation task, add one.
                self._add_eval_task()
            elif self.evluate_tasks.completed_num == 1:
                # Evaluation tasks is completed, log data and add another one.
                objID = next(self.evluate_tasks.completed())[1]
                total_avg_return = ray.get(objID)
                self._add_eval_task()

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
                    sum(
                        ray.get(
                            [
                                sampler.get_total_sample_number.remote()
                                for sampler in self.samplers
                            ]
                        )
                    ),
                )

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

    def _add_eval_task(self):
        self.evaluator.load_state_dict.remote(self.networks.state_dict())
        self.evluate_tasks.add(
            self.evaluator,
            self.evaluator.run_evaluation.remote(self.iteration)
        )
        self.last_eval_iteration = self.iteration


def concate(samples):
    all_samples = {}
    for key in samples[0].keys():
        if samples[0][key] is not None:
            all_samples[key] = torch.cat([sample[key] for sample in samples], dim=0)
    return all_samples
