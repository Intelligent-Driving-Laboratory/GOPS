#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: reference trajectory for model environment
#  Update: 2022-11-16, Yujie Yang: create reference trajectory

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from gops.env.env_ocp.resources.ref_traj_data import (
    DEFAULT_PATH_PARAM,
    DEFAULT_SPEED_PARAM,
)


class MultiRefTrajModel:
    def __init__(
        self,
        path_param: Optional[Dict[str, Dict]] = None,
        speed_param: Optional[Dict[str, Dict]] = None,
    ):
        self.path_param = deepcopy(DEFAULT_PATH_PARAM)
        if path_param is not None:
            for k, v in path_param.items():
                self.path_param[k].update(v)

        self.speed_param = deepcopy(DEFAULT_SPEED_PARAM)
        if speed_param is not None:
            for k, v in speed_param.items():
                self.speed_param[k].update(v)

        ref_speeds = [
            SineRefSpeedModel(**self.speed_param["sine"]),
            ConstantRefSpeedModel(**self.speed_param["constant"]),
        ]

        self.ref_trajs: Sequence[RefTrajModel] = [
            SineRefTrajModel(ref_speeds, **self.path_param["sine"]),
            DoubleLaneRefTrajModel(ref_speeds, **self.path_param["double_lane"]),
            TriangleRefTrajModel(ref_speeds, **self.path_param["triangle"]),
            CircleRefTrajModel(ref_speeds, **self.path_param["circle"]),
        ]

    def compute_x(
        self, t: torch.Tensor, path_num: torch.Tensor, speed_num: torch.Tensor
    ) -> torch.Tensor:
        x = torch.zeros_like(t)
        for i, ref_traj in enumerate(self.ref_trajs):
            x = x + (path_num == i) * ref_traj.compute_x(t, speed_num)
        return x

    def compute_y(
        self, t: torch.Tensor, path_num: torch.Tensor, speed_num: torch.Tensor
    ) -> torch.Tensor:
        y = torch.zeros_like(t)
        for i, ref_traj in enumerate(self.ref_trajs):
            y = y + (path_num == i) * ref_traj.compute_y(t, speed_num)
        return y

    def compute_u(
        self, t: torch.Tensor, path_num: torch.Tensor, speed_num: torch.Tensor
    ) -> torch.Tensor:
        u = torch.zeros_like(t)
        for i, ref_traj in enumerate(self.ref_trajs):
            u = u + (path_num == i) * ref_traj.compute_u(t, speed_num)
        return u

    def compute_phi(
        self, t: torch.Tensor, path_num: torch.Tensor, speed_num: torch.Tensor
    ) -> torch.Tensor:
        phi = torch.zeros_like(t)
        for i, ref_traj in enumerate(self.ref_trajs):
            phi = phi + (path_num == i) * ref_traj.compute_phi(t, speed_num)
        return phi


class RefSpeedModel(metaclass=ABCMeta):
    @abstractmethod
    def compute_u(self, t: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def compute_integrate_u(self, t: torch.Tensor) -> torch.Tensor:
        ...


@dataclass
class ConstantRefSpeedModel(RefSpeedModel):
    u: float

    def compute_u(self, t: torch.Tensor) -> torch.Tensor:
        return self.u * torch.ones_like(t)

    def compute_integrate_u(self, t: torch.Tensor) -> torch.Tensor:
        return self.u * t


@dataclass
class SineRefSpeedModel(RefSpeedModel):
    A: float
    omega: float
    phi: float
    b: float

    def compute_u(self, t: torch.Tensor) -> torch.Tensor:
        return self.A * torch.sin(self.omega * t + self.phi) + self.b

    def compute_integrate_u(self, t: torch.Tensor) -> torch.Tensor:
        return (
            -self.A / self.omega * torch.cos(self.omega * t + self.phi)
            + self.b * t
            + self.A / self.omega * np.cos(self.phi)
        )


@dataclass
class RefTrajModel(metaclass=ABCMeta):
    ref_speeds: Sequence[RefSpeedModel]

    @abstractmethod
    def compute_x(self, t: torch.Tensor, speed_num: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def compute_y(self, t: torch.Tensor, speed_num: torch.Tensor) -> torch.Tensor:
        ...

    def compute_u(self, t: torch.Tensor, speed_num: torch.Tensor) -> torch.Tensor:
        u = torch.zeros_like(t)
        for i, ref_speed in enumerate(self.ref_speeds):
            u = u + (speed_num == i) * ref_speed.compute_u(t)
        return u

    def compute_phi(self, t: torch.Tensor, speed_num: torch.Tensor) -> torch.Tensor:
        dt = 0.001
        dx = self.compute_x(t + dt, speed_num) - self.compute_x(t, speed_num)
        dy = self.compute_y(t + dt, speed_num) - self.compute_y(t, speed_num)
        return torch.atan2(dy, dx)


@dataclass
class SineRefTrajModel(RefTrajModel):
    A: float
    omega: float
    phi: float

    def compute_x(self, t: torch.Tensor, speed_num: torch.Tensor) -> torch.Tensor:
        x = torch.zeros_like(t)
        for i, ref_speed in enumerate(self.ref_speeds):
            x = x + (speed_num == i) * ref_speed.compute_integrate_u(t)
        return x

    def compute_y(self, t: torch.Tensor, speed_num: torch.Tensor) -> torch.Tensor:
        return self.A * torch.sin(self.omega * t + self.phi)


@dataclass
class DoubleLaneRefTrajModel(RefTrajModel):
    t1: float
    t2: float
    t3: float
    t4: float
    y1: float
    y2: float

    def compute_x(self, t: torch.Tensor, speed_num: torch.Tensor) -> torch.Tensor:
        x = torch.zeros_like(t)
        for i, ref_speed in enumerate(self.ref_speeds):
            x = x + (speed_num == i) * ref_speed.compute_integrate_u(t)
        return x

    def compute_y(self, t: torch.Tensor, speed_num: torch.Tensor) -> torch.Tensor:
        y1 = self.y1
        mask1 = t <= self.t1
        y2 = (self.y2 - self.y1) / (self.t2 - self.t1) * (t - self.t1) + self.y1
        mask2 = (t > self.t1) & (t <= self.t2)
        y3 = self.y2
        mask3 = (t > self.t2) & (t <= self.t3)
        y4 = (self.y1 - self.y2) / (self.t4 - self.t3) * (t - self.t3) + self.y2
        mask4 = (t > self.t3) & (t <= self.t4)
        y5 = self.y1
        mask5 = t > self.t4
        y = y1 * mask1 + y2 * mask2 + y3 * mask3 + y4 * mask4 + y5 * mask5
        return y


@dataclass
class TriangleRefTrajModel(RefTrajModel):
    A: float
    T: float

    def compute_x(self, t: torch.Tensor, speed_num: torch.Tensor) -> torch.Tensor:
        x = torch.zeros_like(t)
        for i, ref_speed in enumerate(self.ref_speeds):
            x = x + (speed_num == i) * ref_speed.compute_integrate_u(t)
        return x

    def compute_y(self, t: torch.Tensor, speed_num: torch.Tensor) -> torch.Tensor:
        s = torch.remainder(t, self.T)
        y1 = 2 * self.A / self.T * s
        mask1 = s <= self.T / 2
        y2 = -2 * self.A / self.T * (s - self.T)
        mask2 = (s > self.T / 2) & (s < self.T)
        y = y1 * mask1 + y2 * mask2
        return y


@dataclass
class CircleRefTrajModel(RefTrajModel):
    r: float

    def compute_x(self, t: torch.Tensor, speed_num: torch.Tensor) -> torch.Tensor:
        arc_len = torch.zeros_like(t)
        for i, ref_speed in enumerate(self.ref_speeds):
            arc_len = arc_len + (speed_num == i) * ref_speed.compute_integrate_u(t)
        return self.r * torch.sin(arc_len / self.r)

    def compute_y(self, t: torch.Tensor, speed_num: torch.Tensor) -> torch.Tensor:
        arc_len = torch.zeros_like(t)
        for i, ref_speed in enumerate(self.ref_speeds):
            arc_len = arc_len + (speed_num == i) * ref_speed.compute_integrate_u(t)
        return self.r * (torch.cos(arc_len / self.r) - 1)
