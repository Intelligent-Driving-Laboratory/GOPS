#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab


import torch
import numpy as np
from typing import Union

array_or_torch = Union[torch.Tensor, np.ndarray]

V_F = 5.0  # [m/s]


class CarFollowingDynamics2D:
    def __init__(self):
        """
        state = [ve - vf, gap]
        """
        dt = 0.1

        self.A = torch.as_tensor([[1, 0], [-dt, 1]], dtype=torch.float32)
        self.B = torch.as_tensor([[dt], [0]], dtype=torch.float32)

    @staticmethod
    def get_random_s0(batch):
        devi_v = np.random.uniform(-4, 4, [batch, 1])
        gap = np.random.uniform(5, 10, [batch, 1])
        s0 = np.concatenate([devi_v, gap], axis=1)
        return s0

    def f(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x_next = torch.mm(self.A, x.T) + torch.mm(self.B, u.T)
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
        if numpy_flag:
            x_next = x_next.detach().numpy().squeeze(0)
        return x_next

    @staticmethod
    def compute_reward(x_t: array_or_torch, u_t: array_or_torch) -> array_or_torch:
        numpy_flag = False
        if isinstance(x_t, np.ndarray):
            x_t = torch.as_tensor(x_t, dtype=torch.float32).unsqueeze(0)
            u_t = torch.as_tensor(u_t, dtype=torch.float32).unsqueeze(0)
            numpy_flag = True

        devi_v = x_t[:, 0]
        gap = x_t[:, 1]
        ve = devi_v + V_F

        a_t = u_t[:, 0]
        reward = 0.2 * ve - 0.1 * gap - 0.02 * a_t * a_t
        # print(reward.shape)
        if numpy_flag:
            reward = reward[0].item()
        return reward

    @staticmethod
    def compute_cost(x_t: array_or_torch, u_t: array_or_torch):
        numpy_flag = False
        if isinstance(x_t, np.ndarray):
            x_t = torch.as_tensor(x_t, dtype=torch.float32).unsqueeze(0)
            u_t = torch.as_tensor(u_t, dtype=torch.float32).unsqueeze(0)
            numpy_flag = True
        w = torch.as_tensor([[0.0], [1.0]], dtype=torch.float32)
        hx0 = 2 - torch.mm(x_t, w).detach()

        if numpy_flag:
            hx0 = hx0.detach().numpy().squeeze(0)
        return hx0


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dyn_new = CarFollowingDynamics2D()

    x = torch.Tensor([[2.0, 5.0]])
    u = torch.Tensor([[0.0]])

    x_list = []
    r_list = []
    c_list = []

    x_list.append(x)

    for i in range(100):
        u = torch.Tensor([[-1.0]])
        x = dyn_new.prediction(x, u)
        x_list.append(x)
        r = dyn_new.compute_reward(x, u)
        c = dyn_new.compute_cost(x, u)
        r_list.append(r)
        c_list.append(c)

    X1 = torch.cat(x_list)

    # test state
    plt.plot(X1)
    # plt.plot(X2, '--')
    plt.show()

    # test reward
    plt.plot(r_list)
    # plt.plot(r_old_list, '--')
    plt.show()

    # test cost
    plt.plot(c_list)
    # plt.plot(c_old_list, '--')
    plt.show()
