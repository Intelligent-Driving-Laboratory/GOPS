#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Reply buffer
#  Update: 2021.03.05, Shengbo LI (example, can be deleted)



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
        self.obsv_dim = kwargs['obsv_dim']
        self.act_dim = kwargs['action_dim']
        self.max_size = kwargs['buffer_max_size']
        self.buf = {'obs': np.zeros(combined_shape(self.max_size, self.obsv_dim), dtype=np.float32),
                    'obs2': np.zeros(combined_shape(self.max_size, self.obsv_dim), dtype=np.float32),
                    'act': np.zeros(combined_shape(self.max_size, self.act_dim), dtype=np.float32),
                    'rew': np.zeros(self.max_size, dtype=np.float32),
                    'done': np.zeros(self.max_size, dtype=np.float32),
                    'logp': np.zeros(self.max_size, dtype=np.float32)}
        if 'constraint_dim' in kwargs.keys():
            self.con_dim = kwargs['constraint_dim']
            self.buf['con'] = np.zeros(combined_shape(self.max_size, self.con_dim), dtype=np.float32)
        if 'adversary_dim' in kwargs.keys():
            self.advers_dim = kwargs['adversary_dim']
            self.buf['advers'] = np.zeros(combined_shape(self.max_size, self.advers_dim), dtype=np.float32)
        self.ptr, self.size, = 0, 0

    def __len__(self):
        return self.size

    def __get_RAM__(self):
        return int(sys.getsizeof(self.buf)) * self.size / (self.max_size * 1000000)

    def store(self, obs, act, rew, next_obs, done, logp, time_limited, con=None, advers=None):
        self.buf['obs'][self.ptr] = obs
        self.buf['obs2'][self.ptr] = next_obs
        self.buf['act'][self.ptr] = act
        self.buf['rew'][self.ptr] = rew
        self.buf['done'][self.ptr] = done
        self.buf['logp'][self.ptr] = logp
        if con is not None and 'con' in self.buf.keys():
            self.buf['con'][self.ptr] = con
        if advers is not None and 'advers' in self.buf.keys():
            self.buf['advers'][self.ptr] = advers
        self.ptr = (self.ptr + 1) % self.max_size  # 控制buffer的内存
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, samples):
        for sample in samples:
            self.store(*sample)

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = {}
        for k, v in self.buf.items():
            batch[k] = v[idxs]
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}
