#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Prioritized Replay buffer
#  Update: 2021-05-05, Yuheng Lei: Create prioritized replay buffer
#  Update: 2023-08-08, Zhilong Zheng: Make this compatible with new version of GOPS; Speed up sampling and updating


from typing import Tuple
import numpy as np
import torch
from gops.trainer.buffer.replay_buffer import ReplayBuffer

__all__ = ["PrioritizedReplayBuffer"]


class PrioritizedReplayBuffer(ReplayBuffer):
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

    def __init__(self, index=0, **kwargs):
        super().__init__(index, **kwargs)

        self.sum_tree = np.zeros(2 * self.max_size - 1)
        self.min_tree = float("inf") * np.ones(2 * self.max_size - 1)
        self.alpha = 0.6  #TODO: make it specifiable?
        self.beta = 0.4
        self.beta_increment = 0.01
        self.epsilon = 1e-6
        self.max_priority = 1.0 ** self.alpha

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
        tree_idx = self.ptr + self.max_size - 1
        self.sum_tree[tree_idx] = self.max_priority
        self.min_tree[tree_idx] = self.max_priority
        self.update_tree(tree_idx)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def update_tree(self, tree_idx: int) -> None:
        parent = (tree_idx - 1) // 2
        while True:
            left = 2 * parent + 1
            right = left + 1
            self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
            self.min_tree[parent] = min(self.min_tree[left], self.min_tree[right])
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def get_leaf(self, value: float) -> Tuple[int, float]:
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

    def sample_batch(self, batch_size: int) -> dict:
        idxes, weights = np.zeros(batch_size, dtype=np.int32), np.zeros(batch_size)
        segment = self.sum_tree[0] / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)  #TODO: technically useless
        min_prob = self.min_tree[0] / self.sum_tree[0]
        max_weight = (min_prob * self.size) ** (-self.beta)

        values = np.random.uniform(np.arange(batch_size) * segment, np.arange(batch_size) * segment + segment)
        idxes, priorities = zip(*map(self.get_leaf, values))
        idxes = np.array(idxes, dtype=np.int32)
        priorities = np.array(priorities)
        probs = priorities / self.sum_tree[0]
        weights = (probs * self.size) ** (-self.beta) / max_weight

        batch = {}
        ptrs = idxes - self.max_size + 1
        batch["idx"] = torch.as_tensor(idxes, dtype=torch.int32)
        batch["weight"] = torch.as_tensor(weights, dtype=torch.float32)
        for k, v in self.buf.items():
            if isinstance(v, np.ndarray):
                batch[k] = torch.as_tensor(v[ptrs], dtype=torch.float32)
            else:
                batch[k] = v[ptrs].array2tensor()
        return batch

    def update_batch(self, idxes: int, priorities: float) -> None:
        if isinstance(idxes, torch.Tensor):
            idxes = idxes.detach().numpy()
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().numpy()
        priorities = (priorities + self.epsilon) ** self.alpha
        self.sum_tree[idxes] = priorities
        self.min_tree[idxes] = priorities
        self.max_priority = max(self.max_priority, priorities.max())

        idxes_to_update = {}  # lazy update to avoid redundancy
        for idx in idxes:
            while idx >= 0 and idx not in idxes_to_update:
                idxes_to_update[idx] = True
                idx = (idx - 1) // 2
        for idx in sorted(idxes_to_update.keys(), reverse=True):
            parent = (idx - 1) // 2
            left = 2 * parent + 1
            right = left + 1
            self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
            self.min_tree[parent] = min(self.min_tree[left], self.min_tree[right])

