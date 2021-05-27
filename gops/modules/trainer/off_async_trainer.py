#   Copyright (c) 2020 ocp-tools Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Author: Yang GUAN, Wenhan CAO

__all__ = ['OffAsyncTrainer']

import logging
import queue
import random
import threading

import ray
import torch
from torch.utils.tensorboard import SummaryWriter

from modules.utils.task_pool import TaskPool
from modules.utils.tensorboard_tools import add_scalars
from modules.utils.utils import random_choice_with_index

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from modules.utils.tensorboard_tools import tb_tags
import warnings

warnings.filterwarnings('ignore')


class OffAsyncTrainer():
    def __init__(self, alg, sampler, buffer, evaluator, **kwargs):
        self.algs = alg
        self.samplers = sampler
        self.buffers = buffer
        self.evaluator = evaluator
        self.alg_queue_max_size = kwargs['alg_queue_max_size']
        self.alg_queue = queue.Queue(kwargs['alg_queue_max_size'])
        self.iteration = 0
        self.replay_batch_size = kwargs['replay_batch_size']
        self.max_iteration = kwargs['max_iteration']
        self.ini_network_dir = kwargs['ini_network_dir']

        # create center network
        alg_name = kwargs['algorithm']
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        ApproxContainer = getattr(file, 'ApproxContainer')
        self.networks = ApproxContainer(**kwargs)

        self.ini_network_dir = kwargs['ini_network_dir']

        # initialize the networks
        if self.ini_network_dir is not None:
            self.networks.load_state_dict(torch.load(self.ini_network_dir))

        # start the update thread
        self.update_thread = UpdateThread(self.networks, self.evaluator, **kwargs)
        self.update_thread.start()

        # create sample tasks and pre sampling
        self.sample_tasks = TaskPool()
        self._set_samplers()

        self.warm_size = kwargs['buffer_warm_size']
        while not all([l >= self.warm_size for l in
                       ray.get([rb.__len__.remote() for rb in self.buffers])]):
            for sampler, objID in list(
                    self.sample_tasks.completed()):  # sample_tasks.completed()完成了的sampler任务列表，work进程的名字，objID是进程执行任务的ID
                batch_data = ray.get(objID)  # 得到任务的函数返回值
                random.choice(self.buffers).add_batch.remote(batch_data)  # 随机选择一个buffer，把数据填进去
                self.sample_tasks.add(sampler, sampler.sample.remote())  # 让已经完成了的空闲进程再加进去

        # create buffer tasks and start replay
        self.replay_tasks = TaskPool()  # 创建buffer的任务管理的类
        self._set_buffers()

        # create alg tasks and start computing gradient
        self.learn_tasks = TaskPool()  # 创建learner的任务管理的类
        self._set_algs()

    def _set_samplers(self):
        weights = self.networks.state_dict()  # 得到中心网络参数
        for sampler in self.samplers:  # 对每个sampler进行参数同步
            sampler.load_state_dict.remote(weights)
            self.sample_tasks.add(sampler, sampler.sample.remote())

    def _set_buffers(self):
        for buffer in self.buffers:
            self.replay_tasks.add(buffer, buffer.sample_batch.remote())

    def _set_algs(self):
        weights = self.networks.state_dict()  # 获得中心网络参数
        for alg in self.algs:
            alg.load_state_dict.remote(weights)  # 每个learner同步参数
            buffer, _ = random_choice_with_index(self.buffers)  # 随机选择一个buffer从中采样
            data = ray.get(buffer.sample_batch.remote())  # 得到buffer的采样结果
            self.learn_tasks.add(alg, alg.compute_gradient.remote(data))  # 用采样结果给learner添加计算梯度的任务

    def step(self):
        # sampling
        for sampler, objID in self.sample_tasks.completed():  # 对每个完成的sampler，
            batch_data = ray.get(objID)  # 获得sample的batch
            random.choice(self.buffers).add_batch.remote(batch_data)  # 随机选择buffer，加入batch
            weights = ray.put(self.networks.state_dict())  # 把中心网络的参数放在底层内存里面
            sampler.load_state_dict.remote(weights)  # 同步sampler的参数
            self.sample_tasks.add(sampler, sampler.sample.remote())

        # replay
        for buffer, objID in self.replay_tasks.completed():
            data = ray.get(objID)
            if not self.alg_queue.full():
                self.alg_queue.put(data)  # 储存从buffer里面sample数据的队列
            self.replay_tasks.add(buffer, buffer.sample_batch.remote(self.replay_batch_size))

        # learning
        for alg, objID in self.learn_tasks.completed():
            grads, alg_tb_dict = ray.get(objID)
            if not self.update_thread.inqueue.full():
                self.update_thread.inqueue.put((grads, alg_tb_dict))  # 把梯度放到线程的queue里面来更新中心网络的参数
            data = self.alg_queue.get(block=True)  # 拿到队列里面拿到从buffer采样出来的数据
            weights = ray.put(self.networks.state_dict())  # 把中心网络的参数放在底层内存里面
            alg.load_state_dict.remote(weights)  # 更新learner参数
            self.learn_tasks.add(alg, alg.compute_gradient.remote(data))  # 将完成了的learner重新算梯度

        self.iteration = self.update_thread.iteration

    def train(self):
        while self.iteration < self.max_iteration:
            self.step()


class UpdateThread(threading.Thread):
    def __init__(self, networks, evaluator, **kwargs):  # 需要传进去的参数：中心网络，evaluator，args
        threading.Thread.__init__(self)
        self.networks = networks
        self.evaluator = evaluator
        self.inqueue = queue.Queue(maxsize=20)
        self.stopped = False
        self.save_folder = kwargs['save_folder']
        self.log_save_interval = kwargs['log_save_interval']
        self.apprfunc_save_interval = kwargs['apprfunc_save_interval']
        self.iteration = 0

        self.save_folder = kwargs['save_folder']
        self.log_save_interval = kwargs['log_save_interval']
        self.apprfunc_save_interval = kwargs['apprfunc_save_interval']
        self.eval_interval = kwargs['eval_interval']
        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)
        self.writer.add_scalar(tb_tags['time'], 0, 0)

        self.writer.flush()

    def run(self):
        while not self.stopped:
            self.step()

    def step(self):
        grads, alg_tb_dict = self.inqueue.get(block=True, timeout=30)
        self.networks.update(grads, self.iteration)

        # log
        if self.iteration % self.log_save_interval == 0:
            print('Iter = ', self.iteration)
            add_scalars(alg_tb_dict, self.writer, step=self.iteration)

        # evaluate
        if self.iteration % self.eval_interval == 0:
            self.evaluator.load_state_dict.remote(self.networks.state_dict())
            self.writer.add_scalar(tb_tags['total_average_return'],
                                   ray.get(self.evaluator.run_evaluation.remote(self.iteration)),
                                   self.iteration)

        # save
        if self.iteration % self.apprfunc_save_interval == 0:
            torch.save(self.networks.state_dict(),
                       self.save_folder + '/apprfunc/apprfunc_{}.pkl'.format(self.iteration))

        self.iteration += 1
