#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: reference trajectory for data environment
#  Update: 2022-11-16, Yujie Yang: create reference trajectory

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

DEFAULT_PATH_PARAM = {
    "sine": {"A": 1.5, "omega": 2 * np.pi / 10, "phi": 0.0,},
    "double_lane": {
        "t1": 5.0,
        "t2": 9.0,
        "t3": 14.0,
        "t4": 18.0,
        "y1": 0.0,
        "y2": 3.5,
    },
    "triangle": {"A": 3.0, "T": 10.0,},
    "circle": {"r": 100.0,},
    "straight_lane": {"A": 0.0, "T": 100.0,}
}

DEFAULT_SPEED_PARAM = {
    "sine": {"A": 1.0, "omega": 2 * np.pi / 10, "phi": 0.0, "b": 5.0,},
    "constant": {"u": 5.0,},
}


class MultiRefTrajData:
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
            SineRefSpeedData(**self.speed_param["sine"]),
            ConstantRefSpeedData(**self.speed_param["constant"]),
        ]

        self.ref_trajs: Sequence[RefTrajData] = [
            SineRefTrajData(ref_speeds, **self.path_param["sine"]),
            DoubleLaneRefTrajData(ref_speeds, **self.path_param["double_lane"]),
            TriangleRefTrajData(ref_speeds, **self.path_param["triangle"]),
            CircleRefTrajData(ref_speeds, **self.path_param["circle"]),
            TriangleRefTrajData(ref_speeds, **self.path_param["straight_lane"]),
        ]

    def compute_x(self, t: float, path_num: int, speed_num: int) -> float:
        return self.ref_trajs[path_num].compute_x(t, speed_num)

    def compute_y(self, t: float, path_num: int, speed_num: int) -> float:
        return self.ref_trajs[path_num].compute_y(t, speed_num)

    def compute_u(self, t: float, path_num: int, speed_num: int) -> float:
        return self.ref_trajs[path_num].compute_u(t, speed_num)

    def compute_phi(self, t: float, path_num: int, speed_num: int) -> float:
        return self.ref_trajs[path_num].compute_phi(t, speed_num)


class RefSpeedData(metaclass=ABCMeta):
    @abstractmethod
    def compute_u(self, t: float) -> float:
        ...

    @abstractmethod
    def compute_integrate_u(self, t: float) -> float:
        ...


@dataclass
class ConstantRefSpeedData(RefSpeedData):
    u: float

    def compute_u(self, t: float) -> float:
        return self.u

    def compute_integrate_u(self, t: float) -> float:
        return self.u * t


@dataclass
class SineRefSpeedData(RefSpeedData):
    A: float
    omega: float
    phi: float
    b: float

    def compute_u(self, t: float) -> float:
        return self.A * np.sin(self.omega * t + self.phi) + self.b

    def compute_integrate_u(self, t: float) -> float:
        return (
            -self.A / self.omega * np.cos(self.omega * t + self.phi)
            + self.b * t
            + self.A / self.omega * np.cos(self.phi)
        )


@dataclass
class RefTrajData(metaclass=ABCMeta):
    ref_speeds: Sequence[RefSpeedData]

    @abstractmethod
    def compute_x(self, t: float, speed_num: int) -> float:
        ...

    @abstractmethod
    def compute_y(self, t: float, speed_num: int) -> float:
        ...

    def compute_u(self, t: float, speed_num: int) -> float:
        return self.ref_speeds[speed_num].compute_u(t)

    def compute_phi(self, t: float, speed_num: int) -> float:
        dt = 0.001
        dx = self.compute_x(t + dt, speed_num) - self.compute_x(t, speed_num)
        dy = self.compute_y(t + dt, speed_num) - self.compute_y(t, speed_num)
        return np.arctan2(dy, dx)


@dataclass
class SineRefTrajData(RefTrajData):
    A: float
    omega: float
    phi: float

    def compute_x(self, t: float, speed_num: int) -> float:
        return self.ref_speeds[speed_num].compute_integrate_u(t)

    def compute_y(self, t: float, speed_num: int) -> float:
        return self.A * np.sin(self.omega * t + self.phi)


@dataclass
class DoubleLaneRefTrajData(RefTrajData):
    t1: float
    t2: float
    t3: float
    t4: float
    y1: float
    y2: float

    def compute_x(self, t: float, speed_num: int) -> float:
        return self.ref_speeds[speed_num].compute_integrate_u(t)

    def compute_y(self, t: float, speed_num: int) -> float:
        if t <= self.t1:
            y = self.y1
        elif t <= self.t2:
            k = (self.y2 - self.y1) / (self.t2 - self.t1)
            y = k * (t - self.t1) + self.y1
        elif t <= self.t3:
            y = self.y2
        elif t <= self.t4:
            k = (self.y1 - self.y2) / (self.t4 - self.t3)
            y = k * (t - self.t3) + self.y2
        else:
            y = self.y1
        return y


@dataclass
class TriangleRefTrajData(RefTrajData):
    A: float
    T: float

    def compute_x(self, t: float, speed_num: int) -> float:
        return self.ref_speeds[speed_num].compute_integrate_u(t)

    def compute_y(self, t: float, speed_num: int) -> float:
        s = t % self.T
        if s <= self.T / 2:
            y = 2 * self.A / self.T * s
        else:
            y = -2 * self.A / self.T * (s - self.T)
        return y


@dataclass
class CircleRefTrajData(RefTrajData):
    r: float

    def compute_x(self, t: float, speed_num: int) -> float:
        arc_len = self.ref_speeds[speed_num].compute_integrate_u(t)
        return self.r * np.sin(arc_len / self.r)

    def compute_y(self, t: float, speed_num: int) -> float:
        arc_len = self.ref_speeds[speed_num].compute_integrate_u(t)
        return self.r * (np.cos(arc_len / self.r) - 1)
