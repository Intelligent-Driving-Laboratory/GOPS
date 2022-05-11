#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Serial trainer for RL algorithms
#  Update Date: 2021-03-10, Wenhan CAO: Revise Codes
#  Update Date: 2021-05-21, Shengbo LI: Format Revise


__all__ = ["OnSerialTrainer"]

import logging

import torch
from torch.utils.tensorboard import SummaryWriter

from gops.utils.tensorboard_tools import add_scalars
from gops.utils.utils import ModuleOnDevice

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from gops.utils.tensorboard_tools import tb_tags
import time
import numpy as np
import matplotlib.pyplot as plt


class OnSerialTrainer:
    def __init__(self, alg, sampler, evaluator, **kwargs):
        self.alg = alg
        self.sampler = sampler
        self.evaluator = evaluator

        self.networks = self.alg.networks
        self.sampler.networks = self.networks
        self.evaluator.networks = self.networks

        # Import algorithm, appr func, sampler & buffer
        self.iteration = 0
        self.max_iteration = kwargs.get("max_iteration")
        self.ini_network_dir = kwargs["ini_network_dir"]

        # initialize the networks
        if self.ini_network_dir is not None:
            self.networks.load_state_dict(torch.load(self.ini_network_dir))

        self.save_folder = kwargs["save_folder"]
        self.log_save_interval = kwargs["log_save_interval"]
        self.apprfunc_save_interval = kwargs["apprfunc_save_interval"]
        self.eval_interval = kwargs["eval_interval"]
        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)
        self.writer.add_scalar(tb_tags["alg_time"], 0, 0)
        self.writer.add_scalar(tb_tags["sampler_time"], 0, 0)

        self.writer.flush()
        self.start_time = time.time()
        # setattr(self.alg, "writer", self.evaluator.writer)

        self.use_gpu = kwargs["use_gpu"]

    def step(self):
        # sampling
        (samples_with_replay_format, sampler_tb_dict,) = self.sampler.sample_with_replay_format()
        # learning
        if self.use_gpu:
            for k, v in samples_with_replay_format.items():
                samples_with_replay_format[k] = v.cuda()
        with ModuleOnDevice(self.networks, 'cuda' if self.use_gpu else 'cpu'):
            alg_tb_dict = self.alg.local_update(samples_with_replay_format, self.iteration)

        # log
        if self.iteration % self.log_save_interval == 0:
            print("Iter = ", self.iteration)
            add_scalars(alg_tb_dict, self.writer, step=self.iteration)
            add_scalars(sampler_tb_dict, self.writer, step=self.iteration)
        # evaluate
        if self.iteration % self.eval_interval == 0:
            self.sampler.env.close()
            total_avg_return = self.evaluator.run_evaluation(self.iteration)
            self.evaluator.env.close()
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
            torch.save(
                self.networks.state_dict(),
                self.save_folder + "/apprfunc/apprfunc_{}.pkl".format(self.iteration),
            )
        if self.iteration == self.max_iteration - 1:
            torch.save(
                self.alg.networks.state_dict(),
                self.save_folder + "/apprfunc/apprfunc_{}.pkl".format(self.iteration),
            )
            # A = np.array([[0.4411, -0.6398, 0, 0],
            #               [0.0242, 0.2188, 0, 0],
            #               [0.0703, 0.0171, 1, 2],
            #               [0.0018, 0.0523, 0, 1]])
            # B = np.array([[2.0350], [4.8124], [0.4046], [0.2952]])
            # A = torch.from_numpy(A.astype("float32"))
            # B = torch.from_numpy(B.astype("float32"))
            # obs = torch.Tensor(samples_with_replay_format["obs"])
            # delta_yss = []
            # delta_phiss = []
            # for i in range(100):
            #     delta_yss.append(obs.detach().numpy()[2])
            #     delta_phiss.append(obs.detach().numpy()[3])
            #     v_y, r, delta_y, delta_phi = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
            #     lists_to_stack = [v_y * 1, r * 2, delta_y * 1, delta_phi * 2.4]
            #     scale_obs = torch.stack(lists_to_stack, 1)
            #     steer_norm = self.alg.networks.policy(scale_obs)
            #     actions = steer_norm * 1.2 * np.pi / 9
            #     next_state = [
            #         v_y * A[0, 0] + r * A[0, 1] + delta_y * A[0, 2] + delta_phi * A[0, 3] + B[0, 0] * actions[:, 0],
            #         v_y * A[1, 0] + r * A[1, 1] + delta_y * A[1, 2] + delta_phi * A[1, 3] + B[1, 0] * actions[:, 0],
            #         v_y * A[2, 0] + r * A[2, 1] + delta_y * A[2, 2] + delta_phi * A[2, 3] + B[2, 0] * actions[:, 0],
            #         v_y * A[3, 0] + r * A[3, 1] + delta_y * A[3, 2] + delta_phi * A[3, 3] + B[3, 0] * actions[:, 0]]
            #     next_state = torch.stack(next_state, 1)
            #     v_ys, rs, delta_ys, delta_phis = next_state[:, 0], next_state[:, 1], next_state[:, 2], next_state[:, 3]
            #     delta_phis = torch.where(delta_phis > np.pi, delta_phis - 2 * np.pi, delta_phis)
            #     delta_phis = torch.where(delta_phis <= -np.pi, delta_phis + 2 * np.pi, delta_phis)
            #     obs = torch.stack([v_ys, rs, delta_ys, delta_phis], 1)
            # plt.plot(delta_yss)
            # plt.plot(delta_phiss)
            # plt.show()

    def train(self):

        while self.iteration < self.max_iteration:
            self.step()
            self.iteration += 1

        self.writer.flush()
