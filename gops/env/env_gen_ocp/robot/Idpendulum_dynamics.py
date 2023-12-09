from gym import spaces
import numpy as np

from gops.env.env_gen_ocp.pyth_base import Robot


class IdpendulumParam():
    mass_cart   : float = 9.42477796
    mass_rod1   : float = 4.1033127
    mass_rod2   : float = 4.1033127
    l_rod1      : float = 0.6
    l_rod2      : float = 0.6
    g           : float = 9.81
    damping_cart: float = 0.0
    damping_rod1: float = 0.0
    damping_rod2: float = 0.0


class Dynamics(Robot):
    def __init__(self) -> None:
        self.param = IdpendulumParam()
        self.state=None
        self.state_space=spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
        self.action_space=spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)  
        self.dt = 0.01
        self.discrete_num = 5
        self.max_episode_steps = 500

    def step(self, action: np.ndarray) -> np.ndarray:
        act_batch = 500 * action.reshape(1, -1)
        for _ in range(self.discrete_num):
            self._step(act_batch)
        return self.state

    def _step(self, action: np.ndarray) -> np.ndarray:
        m  = self.param.mass_cart
        m1 = self.param.mass_rod1
        m2 = self.param.mass_rod2
        l1 = self.param.l_rod1
        l2 = self.param.l_rod2
        g  = self.param.g
        d  = self.param.damping_cart
        d1 = self.param.damping_rod1
        d2 = self.param.damping_rod2

        state = self.state.reshape(1, -1)
        p, theta1, theta2, pdot, theta1dot, theta2dot = (
            state[:, 0],
            state[:, 1],
            state[:, 2],
            state[:, 3],
            state[:, 4],
            state[:, 5],
        )
        u = action[:, 0]
        tau = self.dt / self.discrete_num

        ones = np.ones_like(p, dtype=np.float32)
        M = np.stack(
            [
                (m + m1 + m2) * ones,
                l1 * (0.5 * m1 + m2) * np.cos(theta1),
                0.5 * m2 * l2 * np.cos(theta2),
                l1 * (0.5 * m1 + m2) * np.cos(theta1),
                l1 * l1 * (0.3333 * m1 + m2) * ones,
                0.5 * l1 * l2 * m2 * np.cos(theta1 - theta2),
                0.5 * l2 * m2 * np.cos(theta2),
                0.5 * l1 * l2 * m2 * np.cos(theta1 - theta2),
                0.3333 * l2 * l2 * m2 * ones,
            ],
            axis=1,
        ).reshape(-1, 3, 3)

        f = np.stack(
            [
                l1 * (0.5 * m1 + m2) * np.square(theta1dot) * np.sin(theta1)
                + 0.5 * m2 * l2 * np.square(theta2dot) * np.sin(theta2)
                - d * pdot
                + u,
                -0.5
                * l1
                * l2
                * m2
                * np.square(theta2dot)
                * np.sin(theta1 - theta2)
                + g * (0.5 * m1 + m2) * l1 * np.sin(theta1)
                - d1 * theta1dot,
                0.5
                * l1
                * l2
                * m2
                * np.square(theta1dot)
                * np.sin(theta1 - theta2)
                + g * 0.5 * l2 * m2 * np.sin(theta2),
            ],
            axis=1,
        ).reshape(-1, 3, 1)

        M_inv = np.linalg.inv(M)
        tmp = np.matmul(M_inv, f).squeeze(-1)

        deriv = np.concatenate([state[:, 3:], tmp], axis=-1)
        next_states = self.state + tau * deriv
        next_p, next_theta1, next_theta2, next_pdot, next_theta1dot, next_theta2dot = (
            next_states[:, 0],
            next_states[:, 1],
            next_states[:, 2],
            next_states[:, 3],
            next_states[:, 4],
            next_states[:, 5],
        )

        next_p = next_p.reshape(-1, 1)
        next_theta1 = next_theta1.reshape(-1, 1)
        next_theta2 = next_theta2.reshape(-1, 1)
        next_pdot = next_pdot.reshape(-1, 1)
        next_theta1dot = next_theta1dot.reshape(-1, 1)
        next_theta2dot = next_theta2dot.reshape(-1, 1)
        next_states = np.concatenate(
            [
                next_p,
                next_theta1,
                next_theta2,
                next_pdot,
                next_theta1dot,
                next_theta2dot,
            ],
            axis=1, 
        )
        self.state = next_states[0]
        return next_states
