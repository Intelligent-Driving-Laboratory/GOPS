from typing import NamedTuple

import numpy as np
from gymnasium import spaces
from gops.env.env_gen_ocp.pyth_base import Robot


class PendulumParam(NamedTuple):
    max_speed: float = 8.0
    max_torque: float = 2.0
    g: float = 10.0
    m: float = 1.0
    l: float = 1.0


class PendulumDynamics(Robot):
    def __init__(self):
        self.param = PendulumParam()
        self.dt = 0.05

        self.action_space = spaces.Box(
            low=-self.param.max_torque, high=self.param.max_torque, shape=(1,), dtype=np.float32
        )
        self.state_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )

    def step(self, action: np.ndarray) -> np.ndarray:
        th, thdot = self.state

        g = self.param.g
        m = self.param.m
        l = self.param.l
        dt = self.dt

        u = np.clip(action, self.action_space.low, self.action_space.high)[0]

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = np.clip(newthdot, -self.param.max_speed, self.param.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot], dtype=np.float32)
        return self.state
