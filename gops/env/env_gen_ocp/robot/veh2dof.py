import math
from typing import NamedTuple

import numpy as np
from gym import spaces
from gops.env.env_gen_ocp.pyth_base import Robot
from gops.utils.math_utils import angle_normalize


class Veh2DoFParam(NamedTuple):
    kf: float = -128915.5  # front wheel cornering stiffness [N/rad]
    kr: float = -85943.6   # rear wheel cornering stiffness [N/rad]
    lf: float = 1.06       # distance from CG to front axle [m]
    lr: float = 1.85       # distance from CG to rear axle [m]
    m:  float = 1412.0     # mass [kg]
    Iz: float = 1536.7     # polar moment of inertia at CG [kg*m^2]
    u:  float = 5.0        # longitudinal speed [m/s]


class Veh2DoF(Robot):
    def __init__(
        self,
        *,
        dt: float = 0.1,
        max_steer: float = math.pi / 6,
    ):
        self.param = Veh2DoFParam()
        self.dt = dt
        self.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-max_steer], dtype=np.float32),
            high=np.array([max_steer], dtype=np.float32),
        )

    def step(self, action: np.ndarray) -> np.ndarray:
        y, phi, v, w = self.state
        steer = action[0]

        kf = self.param.kf
        kr = self.param.kr
        lf = self.param.lf
        lr = self.param.lr
        m  = self.param.m
        Iz = self.param.Iz
        u  = self.param.u
        dt = self.dt

        next_state = self.state.copy()
        next_state[0] = y + dt * (u * np.sin(phi) + v * np.cos(phi))
        next_state[1] = angle_normalize(phi + dt * w)
        next_state[2] = (
            m * v * u + dt * (lf * kf - lr * kr) * w 
            - dt * kf * steer * u - dt * m * u ** 2 * w
        ) / (m * u - dt * (kf + kr))
        next_state[3] = (
            Iz * w * u + dt * (lf * kf - lr * kr) * v 
            - dt * lf * kf * steer * u
        ) / (Iz * u - dt * (lf ** 2 * kf + lr ** 2 * kr))

        self.state = next_state
        return self.state
