#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yang GUAN
#  Description: Reply buffer


import numpy as np
import sys
import torch

__all__ = ['ReplayBuffer']


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer():
    """
    return torch.tensors
    """

    def __init__(self, **kwargs):
        self.obs_dim = kwargs['obsv_dim']
        self.act_dim = kwargs['action_dim']
        self.max_size = kwargs['buffer_max_size']
        self.obs_buf = np.zeros(combined_shape(self.max_size, self.obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(self.max_size, self.obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(self.max_size, self.act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(self.max_size, dtype=np.float32)
        self.done_buf = np.zeros(self.max_size, dtype=np.float32)
        self.logp_buf = np.zeros(self.max_size, dtype=np.float32)
        self.ptr, self.size, = 0, 0

    def __len__(self):
        return self.size

    def __get_RAM__(self):
        # return self.size * (self.obs_dim * 2 + self.act_dim + self.act + 2)
        return int((sys.getsizeof(self.obs_buf) + sys.getsizeof(self.obs2_buf) + sys.getsizeof(
            self.act_buf) + sys.getsizeof(
            self.rew_buf) + sys.getsizeof(self.done_buf) + sys.getsizeof(self.logp_buf)) * self.size / (
                               self.max_size * 1000000))

    def store(self, obs, act, rew, next_obs, done, logp):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.logp_buf[self.ptr] = logp
        self.ptr = (self.ptr + 1) % self.max_size  # 控制buffer的内存
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, samples):
        for sample in samples:
            self.store(*sample)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     logp=self.logp_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}
