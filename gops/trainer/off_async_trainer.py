#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Asynchronous Parallel Trainer for off-policy RL algorithms
#  Update Date: 2021-05-10, Jiaxin Gao: renew parameters
#  Update: 2022-12-05, Wenhan Cao: add annotation

__all__ = ["OffAsyncTrainer"]

from cmath import inf
import importlib
import os
import random
import time
import warnings

import ray
import torch
from torch.utils.tensorboard import SummaryWriter

from gops.utils.common_utils import random_choice_with_index
from gops.utils.parallel_task_manager import TaskPool
from gops.utils.tensorboard_setup import add_scalars, tb_tags
from gops.utils.log_data import LogData

warnings.filterwarnings("ignore")


class OffAsyncTrainer:
    def __init__(self, alg, sampler, buffer, evaluator, **kwargs):
        self.algs = alg
        self.samplers = sampler
        self.buffers = buffer
        self.per_flag = kwargs["buffer_name"] == "prioritized_replay_buffer"
        if self.per_flag and kwargs["num_buffers"] > 1:
            raise RuntimeError(
                "Using multiple prioritized_replay_buffers is not supported!"
            )
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

        self.replay_batch_size = kwargs["replay_batch_size"]
        self.max_iteration = kwargs["max_iteration"]
        self.sample_interval = kwargs.get("sample_interval", 1)
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

        # create sample tasks and pre sampling
        self.sample_tasks = TaskPool()
        self._set_samplers()
        self.sampler_tb_dict = LogData()

        self.warm_size = kwargs["buffer_warm_size"]
        while not all(
            [
                l >= self.warm_size
                for l in ray.get([rb.__len__.remote() for rb in self.buffers])
            ]
        ):
            for sampler, objID in list(self.sample_tasks.completed()):
                batch_data, _ = ray.get(objID)
                random.choice(self.buffers).add_batch.remote(batch_data)
                self.sample_tasks.add(sampler, sampler.sample.remote())

        # create alg tasks and start computing gradient
        self.learn_tasks = TaskPool()
        self._set_algs()

        # create evaluation tasks
        self.evluate_tasks = TaskPool()
        self.last_eval_iteration = 0

        self.use_gpu = kwargs["use_gpu"]
        if self.use_gpu:
            for alg in self.algs:
                alg.to.remote("cuda")

        self.start_time = time.time()

    def _set_samplers(self):
        weights = self.networks.state_dict()
        for sampler in self.samplers:
            sampler.load_state_dict.remote(weights)
            self.sample_tasks.add(sampler, sampler.sample.remote())

    def _set_algs(self):
        weights = self.networks.state_dict()
        for alg in self.algs:
            alg.train.remote()
            alg.load_state_dict.remote(weights)
            buffer, _ = random_choice_with_index(self.buffers)
            data = ray.get(buffer.sample_batch.remote(self.replay_batch_size))
            self.learn_tasks.add(
                alg, alg.get_remote_update_info.remote(data, self.iteration)
            )

    def step(self):
        # sampling
        if self.iteration % self.sample_interval == 0:
            if self.sample_tasks.completed_num > 0:
                weights = ray.put(self.networks.state_dict())
                for sampler, objID in self.sample_tasks.completed():
                    batch_data, sampler_tb_dict = ray.get(objID)
                    random.choice(self.buffers).add_batch.remote(batch_data)
                    sampler.load_state_dict.remote(weights)
                    self.sample_tasks.add(sampler, sampler.sample.remote())
                    self.sampler_tb_dict.add_average(sampler_tb_dict)

        # learning
        for alg, objID in self.learn_tasks.completed():
            if self.per_flag:
                extra_info, update_info = ray.get(objID)
                alg_tb_dict, idx, new_priority = extra_info
                self.buffers[0].update_batch.remote(idx, new_priority)
            else:
                alg_tb_dict, update_info = ray.get(objID)

            # replay
            data = ray.get(
                random.choice(self.buffers).sample_batch.remote(self.replay_batch_size)
            )
            if self.use_gpu:
                for k, v in data.items():
                    data[k] = v.cuda()

            weights = ray.put(self.networks.state_dict())
            alg.load_state_dict.remote(weights)
            self.learn_tasks.add(
                alg, alg.get_remote_update_info.remote(data, self.iteration)
            )
            if self.use_gpu:
                for k, v in update_info.items():
                    if isinstance(v, list):
                        for i in range(len(v)):
                            update_info[k][i] = v[i].cpu()
            self.networks.remote_update(update_info)

            self.iteration += 1

            # log
            if self.iteration % self.log_save_interval == 0:
                print("Iter = ", self.iteration)
                add_scalars(alg_tb_dict, self.writer, step=self.iteration)
                add_scalars(self.sampler_tb_dict.pop(), self.writer, step=self.iteration)

            # save networks
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
                    tb_tags["Buffer RAM of RL iteration"],
                    sum(
                        ray.get(
                            [buffer.__get_RAM__.remote() for buffer in self.buffers]
                        )
                    ),
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
