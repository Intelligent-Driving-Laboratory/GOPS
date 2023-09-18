import math
from typing import NamedTuple

import numpy as np
from gym import spaces
from gops.env.env_gen_ocp.pyth_base import Robot


class Veh3DoFParam(NamedTuple):
    kf: float = -128915.5  # front wheel cornering stiffness [N/rad]
    kr: float = -85943.6   # rear wheel cornering stiffness [N/rad]
    lf: float = 1.06       # distance from CG to front axle [m]
    lr: float = 1.85       # distance from CG to rear axle [m]
    m:  float = 1412.0     # mass [kg]
    Iz: float = 1536.7     # polar moment of inertia at CG [kg*m^2]


class Veh3DoF(Robot):
    def __init__(
        self,
        *,
        dt: float = 0.1,
        max_acc: float = 3.0,
        max_steer: float = math.pi / 6,
    ):
        self.param = Veh3DoFParam()
        self.dt = dt
        self.state = None
        self.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-max_steer, -max_acc], dtype=np.float32),
            high=np.array([max_steer, max_acc], dtype=np.float32),
        )

    def reset(self, state: np.ndarray) -> np.ndarray:
        self.state = state.copy()
        return self.state

    def step(self, action: np.ndarray) -> np.ndarray:
        x, y, phi, u, v, w = self.state
        steer, ax = action

        kf = self.param.kf
        kr = self.param.kr
        lf = self.param.lf
        lr = self.param.lr
        m  = self.param.m
        Iz = self.param.Iz
        dt = self.dt

        next_state = self.state.copy()
        next_state[0] = x + dt * (u * np.cos(phi) - v * np.sin(phi))
        next_state[1] = y + dt * (u * np.sin(phi) + v * np.cos(phi))
        next_state[2] = angle_normalize(phi + dt * w)
        next_state[3] = u + dt * ax
        next_state[4] = (
            m * v * u + dt * (lf * kf - lr * kr) * w 
            - dt * kf * steer * u - dt * m * u ** 2 * w
        ) / (m * u - dt * (kf + kr))
        next_state[5] = (
            Iz * w * u + dt * (lf * kf - lr * kr) * v 
            - dt * lf * kf * steer * u
        ) / (Iz * u - dt * (lf ** 2 * kf + lr ** 2 * kr))

        self.state = next_state
        return self.state


def angle_normalize(x):
    return ((x + math.pi) % (2 * math.pi)) - math.pi
