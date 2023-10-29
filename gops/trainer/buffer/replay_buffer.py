#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Replay buffer
#  Update: 2021-03-05, Yuheng Lei: Create replay buffer


import numpy as np
import sys
import torch
from gops.utils.common_utils import set_seed

__all__ = ["ReplayBuffer"]


def combined_shape(length: int, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer:
    """
    Implementation of replay buffer with uniform sampling probability.
    """

    def __init__(self, index=0, **kwargs):
        set_seed(kwargs["trainer"], kwargs["seed"], index + 100)
        self.obsv_dim = kwargs["obsv_dim"]
        self.act_dim = kwargs["action_dim"]
        self.max_size = kwargs["buffer_max_size"]
        self.buf = {
            "obs": np.zeros(
                combined_shape(self.max_size, self.obsv_dim), dtype=np.float32
            ),
            "obs2": np.zeros(
                combined_shape(self.max_size, self.obsv_dim), dtype=np.float32
            ),
            "act": np.zeros(
                combined_shape(self.max_size, self.act_dim), dtype=np.float32
            ),
            "rew": np.zeros(self.max_size, dtype=np.float32),
            "done": np.zeros(self.max_size, dtype=np.float32),
            "logp": np.zeros(self.max_size, dtype=np.float32),
        }
        self.additional_info = kwargs["additional_info"]
        for k, v in self.additional_info.items():
            if isinstance(v, dict):
                self.buf[k] = np.zeros(
                    combined_shape(self.max_size, v["shape"]), dtype=v["dtype"]
                )
                self.buf["next_" + k] = np.zeros(
                    combined_shape(self.max_size, v["shape"]), dtype=v["dtype"]
                )
            else:
                self.buf[k] = v.batch(self.max_size)
                self.buf["next_" + k] = v.batch(self.max_size)
        self.ptr, self.size, = (
            0,
            0,
        )

    def __len__(self):
        return self.size

    def __get_RAM__(self):
        return int(sys.getsizeof(self.buf)) * self.size / (self.max_size * 1000000)

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        done: bool,
        info: dict,
        next_obs: np.ndarray,
        next_info: dict,
        logp: np.ndarray,
    ) -> None:
        self.buf["obs"][self.ptr] = obs
        self.buf["obs2"][self.ptr] = next_obs
        self.buf["act"][self.ptr] = act
        self.buf["rew"][self.ptr] = rew
        self.buf["done"][self.ptr] = done
        self.buf["logp"][self.ptr] = logp
        for k in self.additional_info.keys():
            self.buf[k][self.ptr] = info[k]
            self.buf["next_" + k][self.ptr] = next_info[k]
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, samples: list) -> None:
        list(map(lambda sample: self.store(*sample), samples))

    def sample_batch(self, batch_size: int) -> dict:
        idxes = np.random.randint(0, self.size, size=batch_size)
        batch = {}
        for k, v in self.buf.items():
            if isinstance(v, np.ndarray):
                batch[k] = torch.as_tensor(v[idxes], dtype=torch.float32)
            else:
                batch[k] = v[idxes].array2tensor()
        return batch
