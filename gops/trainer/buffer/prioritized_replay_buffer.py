#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Replay buffer
#  Update: 2021-05-05, Yuheng Lei: Create prioritized replay buffer


import numpy as np
import sys
import torch

__all__ = ["PrioritizedReplayBuffer"]


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class PrioritizedReplayBuffer(object):
    """
    Implementation of replay buffer with prioritized sampling probability.

    Paper:
        https://openreview.net/forum?id=pBbWjZdoRiN

    Args:
        alpha (float, optional): Determines how much prioritization is used,
                                 with alpha = 0 corresponding to uniform case.
                                 Defaults to 0.6.
        beta (float, optional): Initial strength of compensation for non-uniform probabilities,
                                with beta = 1 corresponding to fully compensation.
                                Defaults to 0.4.
        beta_increment (float, optional): Schedule on beta that finally reaches 1.
                                          Defaults to 0.01.
    """

    def __init__(self, **kwargs):
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
        if "constraint_dim" in kwargs.keys():
            self.con_dim = kwargs["constraint_dim"]
            self.buf["con"] = np.zeros(
                combined_shape(self.max_size, self.con_dim), dtype=np.float32
            )
        if "adversary_dim" in kwargs.keys():
            self.advers_dim = kwargs["adversary_dim"]
            self.buf["advers"] = np.zeros(
                combined_shape(self.max_size, self.advers_dim), dtype=np.float32
            )
        self.ptr, self.size, = (
            0,
            0,
        )
        self.sum_tree = np.zeros(2 * self.max_size - 1)
        self.min_tree = float("inf") * np.ones(2 * self.max_size - 1)
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.01
        self.epsilon = 1e-6
        self.max_priority = 1.0 ** self.alpha

    def __len__(self):
        return self.size

    def __get_RAM__(self):
        return int(sys.getsizeof(self.buf)) * self.size / (self.max_size * 1000000)

    def store(
        self, obs, act, rew, next_obs, done, logp, time_limited, con=None, advers=None
    ):
        self.buf["obs"][self.ptr] = obs
        self.buf["obs2"][self.ptr] = next_obs
        self.buf["act"][self.ptr] = act
        self.buf["rew"][self.ptr] = rew
        self.buf["done"][self.ptr] = done
        self.buf["logp"][self.ptr] = logp
        if con is not None and "con" in self.buf.keys():
            self.buf["con"][self.ptr] = con
        if advers is not None and "advers" in self.buf.keys():
            self.buf["advers"][self.ptr] = advers
        tree_idx = self.ptr + self.max_size - 1
        self.update_tree(tree_idx, self.max_priority)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, samples):
        for sample in samples:
            self.store(*sample)

    def update_tree(self, tree_idx, priority):
        self.sum_tree[tree_idx] = priority
        self.min_tree[tree_idx] = priority
        parent = (tree_idx - 1) // 2
        while True:
            left = 2 * parent + 1
            right = left + 1
            self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
            self.min_tree[parent] = min(self.min_tree[left], self.min_tree[right])
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def get_leaf(self, value):
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.sum_tree):
                idx = parent
                break
            else:
                if value <= self.sum_tree[left]:
                    parent = left
                else:
                    value -= self.sum_tree[left]
                    parent = right
        return idx, self.sum_tree[idx]

    def sample_batch(self, batch_size):
        idxes, weights = np.zeros(batch_size, dtype=np.int), np.zeros(batch_size)
        segment = self.sum_tree[0] / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)
        min_prob = self.min_tree[0] / self.sum_tree[0]
        max_weight = (min_prob * self.size) ** (-self.beta)
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            value = np.random.uniform(a, b)
            idx, priority = self.get_leaf(value)
            prob = priority / self.sum_tree[0]
            weight = np.power(prob * self.size, -self.beta)
            weights[i] = weight / max_weight
            idxes[i] = idx
        batch = {}
        ptrs = idxes - self.max_size + 1
        batch["idx"] = torch.as_tensor(idxes, dtype=torch.int)
        batch["weight"] = torch.as_tensor(weights, dtype=torch.float32)
        for k, v in self.buf.items():
            batch[k] = torch.as_tensor(v[ptrs].copy(), dtype=torch.float32)
        return batch

    def update_batch(self, idxes, priorities):
        for idx, priority in zip(idxes, priorities):
            priority = (priority + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.update_tree(idx, priority)
