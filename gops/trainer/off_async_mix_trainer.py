#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Mixed Asynchronous Parallel Trainer
#  Update Date: 2021-05-10, Jiaxin Gao: create mixed asynchronous trainer


__all__ = ["OffAsyncTrainermix"]

from cmath import inf
import importlib
import os
import random
import time
import warnings

import ray
import torch
from torch.utils.tensorboard import SummaryWriter

from gops.utils.parallel_task_manager import TaskPool
from gops.utils.tensorboard_setup import add_scalars
from gops.utils.tensorboard_setup import tb_tags
from gops.utils.common_utils import random_choice_with_index

warnings.filterwarnings("ignore")


class OffAsyncTrainermix:
    def __init__(self, alg, sampler, buffer, evaluator, **kwargs):
        self.algs = alg
        self.samplers = sampler
        self.buffers = buffer
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

        # initialize the networks
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
        add_scalars({"alg_time": 0, "sampler_time": 0}, self.writer, 0)
        self.writer.flush()

        # create sample tasks and pre sampling
        self.sample_tasks = TaskPool()
        self._set_samplers()

        self.warm_size = kwargs["buffer_warm_size"]
        while not all(
                [
                    l >= self.warm_size
                    for l in ray.get([rb.__len__.remote() for rb in self.buffers])
                ]
        ):
            for sampler, objID in list(
                    self.sample_tasks.completed()
            ):  # sample_tasks.completed()完成了的sampler任务列表，work进程的名字，objID是进程执行任务的ID
                batch_data, _ = ray.get(objID)  # 得到任务的函数返回值
                random.choice(self.buffers).add_batch.remote(
                    batch_data
                )  # 随机选择一个buffer，把数据填进去
                self.sample_tasks.add(
                    sampler, sampler.sample.remote()
                )  # 让已经完成了的空闲进程再加进去

        # create alg tasks and start computing gradient
        self.learn_tasks = TaskPool()  # 创建learner的任务管理的类
        self._set_algs()

        self.start_time = time.time()

    def _set_samplers(self):
        weights = self.networks.state_dict()  # 得到中心网络参数
        for sampler in self.samplers:  # 对每个sampler进行参数同步
            sampler.load_state_dict.remote(weights)
            self.sample_tasks.add(sampler, sampler.sample.remote())

    def _set_algs(self):
        weights = self.networks.state_dict()  # 获得中心网络参数
        for alg in self.algs:
            alg.load_state_dict.remote(weights)  # 每个learner同步参数
            buffer, _ = random_choice_with_index(self.buffers)  # 随机选择一个buffer从中采样
            data = ray.get(
                buffer.sample_batch.remote(self.replay_batch_size)
            )  # 得到buffer的采样结果
            self.learn_tasks.add(
                alg, alg.get_remote_update_info.remote(data, self.iteration)
            )  # 用采样结果给learner添加计算梯度的任务

    def step(self):
        # sampling
        sampler_tb_dict = {}
        if self.iteration % self.sample_interval == 0:
            if self.sample_tasks.completed() is not None:
                weights = ray.put(self.networks.state_dict())  # 把中心网络的参数放在底层内存里面
                for sampler, objID in self.sample_tasks.completed():  # 对每个完成的sampler，
                    batch_data, sampler_tb_dict = ray.get(objID)  # 获得sample的batch
                    random.choice(self.buffers).add_batch.remote(
                        batch_data
                    )  # 随机选择buffer，加入batch
                    sampler.load_state_dict.remote(weights)  # 同步sampler的参数
                    self.sample_tasks.add(sampler, sampler.sample.remote())

        # learning
        for alg, objID in self.learn_tasks.completed():
            alg_tb_dict, update_info = ray.get(objID)
            # replay
            data = random.choice(self.buffers).sample_batch.remote(
                self.replay_batch_size
            )
            alg.remote_update.remote(update_info)  # 更新learner参数
            self.learn_tasks.add(
                alg, alg.get_remote_update_info.remote(data, self.iteration)
            )  # 将完成了的learner重新算梯度
            self.iteration += 1
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
                
                if total_avg_return > self.best_tar:
                    self.best_tar = total_avg_return
                    print('New best TAR = {}!'.format(str(self.best_tar)))

                    for filename in os.listdir(self.save_folder + "/apprfunc/"):
                        if filename.endswith("_opt.pkl"):
                            os.remove(self.save_folder + "/apprfunc/" + filename)
                    
                    torch.save(
                        self.networks.state_dict(),
                        self.save_folder + "/apprfunc/apprfunc_{}_opt.pkl".format(self.iteration),
                    )

                # get ram for buffer
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

            # save
            if self.iteration % self.apprfunc_save_interval == 0:
                torch.save(
                    self.networks.state_dict(),
                    self.save_folder
                    + "/apprfunc/apprfunc_{}.pkl".format(self.iteration),
                )

    def train(self):
        while self.iteration < self.max_iteration:
            self.step()
            i = 0
            for _ in self.algs:
                i += 1
            if i == 1:
                weights = ray.get(self.algs[0].state_dict.remote())
                self.networks.load_state_dict(weights)

            elif i > 1:
                if self.iteration % 50 == 0:
                    weights_last_time = None
                    values_last_time = None
                    for j in range(i):
                        weights = ray.get(self.algs[j].state_dict.remote())  # get para
                        values = list(weights.values())
                        keys = weights.keys()
                        if j == 0:
                            values_last_time = values
                            weights_last_time = dict(zip(keys, values_last_time))
                        else:
                            values_last_time = [a * j for a in values_last_time]
                            values_last_time = [
                                a + b for a, b in zip(values_last_time, values)
                            ]
                            values_last_time = [
                                a / (j + 1) for a in values_last_time
                            ]  # error
                            weights_last_time = dict(zip(keys, values_last_time))

                    self.networks.load_state_dict(weights_last_time)  # load para
                    weights = ray.put(self.networks.state_dict())

                    for alg in self.algs:
                        alg.load_state_dict.remote(weights)
