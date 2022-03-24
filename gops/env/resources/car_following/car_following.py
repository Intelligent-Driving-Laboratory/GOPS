#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Car Following model
#  Update Date: 2021-05-55, Jie Li: car following dynamics


import torch
import numpy as np
from typing import Union

array_or_torch = Union[torch.Tensor, np.ndarray]


class CarFollowingDynamics:
    def __init__(self):

        dt = 0.1
        self.A = torch.as_tensor([[1, 0.0, -0.00], [0.0, 1, 0], [-dt, dt, 1.0]], dtype=torch.float32)
        self.B = torch.as_tensor([[dt], [0], [0]], dtype=torch.float32)
        self.D = torch.as_tensor([[0], [dt], [0]], dtype=torch.float32)
        self.Q = torch.as_tensor([[0.2, 0, 0], [0, 0, 0], [0, 0, -0.1]], dtype=torch.float32)
        self.R = torch.as_tensor([[-0.02]], dtype=torch.float32)

        self.mu = 0.0
        self.var = 0.7
        self.max_d = 7
        self.min_d = -7
        # self.action_rescale = 0.5

    def get_random_s0(self, batch):

        v_e = np.random.uniform(0, 7, [batch, 1])
        v_e = np.clip(v_e, 0, 7)

        v_t = np.random.uniform(2, 7, [batch, 1])
        v_t = np.clip(v_t, 2, 7)

        gap = np.random.uniform(8, 15, [batch, 1])
        gap = np.clip(gap, 1, 20)
        x = np.concatenate([v_e, v_t, gap], axis=1)
        return x

    def f(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        x' = f(x, u)

        Parameters
        ----------
        x: [b,3]
        u: [b,1]

        Returns
        -------
        x_dot: [b,3]
        """
        d = torch.as_tensor(
            np.clip(
                np.random.normal(self.mu, self.var, [1, u.size()[0]]),
                self.min_d,
                self.max_d,
            ),
            dtype=torch.float32,
        )
        # print(d)
        x_next = torch.mm(self.A, x.T) + torch.mm(self.B, u.T) + torch.mm(self.D, d)
        return x_next.T

    def prediction(self, x_t: array_or_torch, u_t: array_or_torch) -> array_or_torch:
        """
        environment dynamics in torch, batch operation,
                        or in numpy, non-batch

        Parameters
        ----------
        x_t: [b,3]
        u_t: [b,1]

        Returns
        -------
        x_next: [b,3]
        """

        numpy_flag = False
        if isinstance(x_t, np.ndarray):
            x_t = torch.as_tensor(x_t, dtype=torch.float32).unsqueeze(0)
            u_t = torch.as_tensor(u_t, dtype=torch.float32).unsqueeze(0)
            numpy_flag = True
        x_next = self.f(x_t, u_t)
        print(x_t, u_t, x_next)
        if numpy_flag:
            x_next = x_next.detach().numpy().squeeze(0)
        return x_next

    def compute_reward(self, x_t: array_or_torch, u_t: array_or_torch) -> array_or_torch:
        """
        reward in torch, batch operation

        Parameters
        ----------
        x_t: [b,3]
        u_t: [b,1]

        Returns
        -------
        reward : [b,3]
        """
        numpy_flag = False
        if isinstance(x_t, np.ndarray):
            x_t = torch.as_tensor(x_t, dtype=torch.float32).unsqueeze(0)
            u_t = torch.as_tensor(u_t, dtype=torch.float32).unsqueeze(0)
            numpy_flag = True
        reward = torch.sum(torch.mm(x_t, self.Q), 1) + torch.sum(torch.mm(u_t, self.R) * u_t, 1)
        # print(reward.shape)
        if numpy_flag:
            reward = reward[0].item()
        return reward

    def compute_cost(self, x_t: array_or_torch, u_t: array_or_torch):
        numpy_flag = False
        if isinstance(x_t, np.ndarray):
            x_t = torch.as_tensor(x_t, dtype=torch.float32).unsqueeze(0)
            u_t = torch.as_tensor(u_t, dtype=torch.float32).unsqueeze(0)
            numpy_flag = True
        w = torch.as_tensor([[0.0], [0.0], [1.0]], dtype=torch.float32)
        w = torch.Tensor([[0], [0], [1]])
        hx0 = 2 - torch.mm(x_t, w).detach()
        # print(x, w)
        # hx1 = 2 - torch.mm(x_next, w)
        if numpy_flag:
            hx0 = hx0.detach().numpy().squeeze(0)
        return hx0


if __name__ == "__main__":
    import ref
    import matplotlib.pyplot as plt

    dyn_new = CarFollowingDynamics()
    dyn_old = ref.StateModel()

    x = torch.Tensor([[2.0, 3.0, 0.0]])
    u = torch.Tensor([[2]])

    x_new_list = []
    x_old_list = []

    r_new_list = []
    r_old_list = []

    c_new_list = []
    c_old_list = []

    x_new_list.append(x)
    x_old_list.append(x)

    x1 = x.clone()
    x2 = x.clone()
    for i in range(100):
        u = torch.randn(1, 1) * 2
        x1 = dyn_new.prediction(x1, u)
        x_new_list.append(x1)
        r1 = dyn_new.compute_reward(x1, u)
        c1 = dyn_new.compute_cost(x1, u)
        r_new_list.append(r1)
        c_new_list.append(c1)

        x2, r2, c, d = dyn_old.step(x2, u)
        x_old_list.append(x2)
        r_old_list.append(r2)
        c_old_list.append(c[2])

    X1 = torch.cat(x_new_list)
    X2 = torch.cat(x_old_list)
    ## test state
    # plt.plot(X1)
    # plt.plot(X2, '--')
    # plt.show()

    # # test reward
    # plt.plot(r_new_list)
    # plt.plot(r_old_list, '--')
    # plt.show()

    # # # test cost
    # plt.plot(c_new_list)
    # plt.plot(c_old_list, '--')
    # plt.show()
