from typing import Optional

from gym import spaces
import numpy as np

from gops.env.env_gen_ocp.pyth_base import Env, State, Robot


class CartpoleParam():
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    total_mass: float = masspole + masscart
    length: float = 0.5
    polemass_length: float = masspole * length
    force_mag: float = 10.0


class Dynamics(Robot):
    theta_threshold_radians = 12 * 2 * np.pi / 360
    x_threshold = 2.4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.param = CartpoleParam()
        self.dt = 0.02
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ]
        )
        self.state_space = spaces.Box(-high, high)
        self.state = None

    def step(self, action: np.ndarray) -> np.ndarray:
        action = np.expand_dims(action, 0)
        force = self.param.force_mag * float(action)

        gravity = self.param.gravity
        masspole = self.param.masspole
        total_mass = self.param.total_mass
        length = self.param.length
        polemass_length = self.param.polemass_length

        x, x_dot, theta, theta_dot = self.state
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (
            force + polemass_length * theta_dot * theta_dot * sintheta
        ) / total_mass
        thetaacc = (gravity * sintheta - costheta * temp) / (
            length
            * (4.0 / 3.0 - masspole * costheta * costheta / total_mass)
        )

        xacc = temp - polemass_length * thetaacc * costheta / total_mass
        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * xacc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot])
        return self.state
        
        