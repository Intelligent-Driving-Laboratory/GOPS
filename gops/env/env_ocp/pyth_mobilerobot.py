#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Mobile Robot Environment
#  Update Date: 2022-06-05, Baiyu Peng: create environment

from typing import Any, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces

from gops.env.env_ocp.pyth_base_env import PythBaseEnv

gym.logger.setLevel(gym.logger.ERROR)


class PythMobilerobot(PythBaseEnv):
    def __init__(
        self, **kwargs: Any,
    ):
        self.n_obstacle = 1
        self.safe_margin = 0.15
        self.max_episode_steps = 200
        work_space = kwargs.pop("work_space", None)
        if work_space is None:
            # initial range of robot state
            robot_high = np.array([2.7, 1, 0.6, 0.3, 0], dtype=np.float32)
            robot_low = np.array([0, -1, -0.6, 0, 0], dtype=np.float32)

            # initial range of tracking error
            error_high = np.zeros(3, dtype=np.float32)
            error_low = np.zeros(3, dtype=np.float32)

            # initial range of obstacle
            obstacle_high = np.array([6, 3, np.pi / 2 + 0.3, 0.5, 0], dtype=np.float32)
            obstacle_low = np.array(
                [3.5, -3, np.pi / 2 - 0.3, 0.0, 0], dtype=np.float32
            )

            init_high = np.concatenate(
                [robot_high, error_high] + [obstacle_high] * self.n_obstacle
            )
            init_low = np.concatenate(
                [robot_low, error_low] + [obstacle_low] * self.n_obstacle
            )
            work_space = np.stack((init_low, init_high))
        super(PythMobilerobot, self).__init__(work_space=work_space, **kwargs)

        self.robot = Robot()
        self.obses = [Robot() for _ in range(self.n_obstacle)]
        self.dt = 0.2
        self.state_dim = (1 + self.n_obstacle) * 5 + 3
        self.action_dim = 2
        self.use_constraint = kwargs.get("use_constraint", True)
        self.constraint_dim = self.n_obstacle

        lb_state = np.array(
            [-30, -30, -np.pi, -1, -np.pi / 2]
            + [-30, -np.pi, -2]
            + [-30, -30, -np.pi, -1, -np.pi / 2] * self.n_obstacle
        )
        hb_state = np.array(
            [60, 30, np.pi, 1, np.pi / 2]
            + [30, np.pi, 2]
            + [30, 30, np.pi, 1, np.pi / 2] * self.n_obstacle
        )
        lb_action = np.array([-0.4, -np.pi / 3])
        hb_action = np.array([0.4, np.pi / 3])

        self.action_space = spaces.Box(low=lb_action, high=hb_action)
        self.observation_space = spaces.Box(lb_state, hb_state)

        self.seed()
        self._state = self.reset()

        self.steps = 0

    @property
    def additional_info(self):
        return {
            "constraint": {"shape": (0,), "dtype": np.float32},
        }

    @property
    def state(self):
        return self._state.reshape(-1)[:5]

    def reset(self, init_state: list = None, **kwargs: Any) -> Tuple[np.ndarray, dict]:
        if init_state is None:
            state = [self.sample_initial_state()]
        else:
            state = [init_state]
        state = np.array(state, dtype=np.float32)
        state[:, 5:8] = self.robot.tracking_error(state[:, :5])
        self.steps_beyond_done = None
        self.steps = 0
        self._state = state

        return self._state.reshape(-1), {"constraint": self.get_constraint()}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, dict]:
        #  define your forward function here: the format is just like: state_next = f(state,action)
        action = action.reshape(1, -1)
        for i in range(1 + self.n_obstacle):
            if i == 0:
                robot_state = self.robot.f_xu(
                    self._state[:, :5], action.reshape(1, -1), self.dt, "ego"
                )
                tracking_error = self.robot.tracking_error(robot_state)
                state_next = np.concatenate((robot_state, tracking_error), 1)

            else:
                obs_state = self.robot.f_xu(
                    self._state[:, 3 + i * 5 : 3 + i * 5 + 5],
                    self._state[:, 3 + i * 5 + 3 : 3 + i * 5 + 5],
                    self.dt,
                    "obs",
                )
                state_next = np.concatenate((state_next, obs_state), 1)

        self._state = state_next

        # define the reward function here the format is just like: reward = l(state,state_next,reward)
        r_tracking = (
            -1.4 * np.square(tracking_error[:, 0])
            - 1 * np.square(tracking_error[:, 1])
            - 16 * np.square(tracking_error[:, 2])
        )
        r_action = -0.2 * np.square(action[:, 0]) - 0.5 * np.square(action[:, 1])
        reward = r_tracking + r_action

        # define the constraint here
        constraint = self.get_constraint()
        dead = constraint > 0

        # define the ending condition here the format is just like isdone = l(next_state)
        isdone = self.get_done()

        self.steps += 1
        info = {"constraint": constraint}
        return (
            np.array(self._state.reshape(-1), dtype=np.float32),
            float(reward),
            isdone,
            info,
        )

    def get_done(self) -> np.ndarray:
        done = self._state[:, 0] < -2 or self._state[:, 1] > 4 or self._state[:, 1] < -4
        for i in range(self.n_obstacle):
            crush = (
                (
                    (
                        (self._state[:, 8 + i * 5] - self._state[:, 0]) ** 2
                        + (self._state[:, 9 + i * 5] - self._state[:, 1]) ** 2
                    )
                )
                ** 0.5
                - (
                    self.robot.robot_params["radius"]
                    + self.obses[i].robot_params["radius"]
                )
                < 0
            )
            done = done or crush
        return done

    def get_constraint(self) -> np.ndarray:
        constraint = np.zeros((self._state.shape[0], self.n_obstacle))
        for i in range(self.n_obstacle):
            safe_dis = (
                self.robot.robot_params["radius"]
                + self.obses[i].robot_params["radius"]
                + self.safe_margin
            )
            constraint[:, i] = (
                safe_dis
                - (
                    (
                        (self._state[:, 8 + i * 5] - self._state[:, 0]) ** 2
                        + (self._state[:, 9 + i * 5] - self._state[:, 1]) ** 2
                    )
                )
                ** 0.5
            )
        return constraint.reshape(-1)

    def render(self, mode: str = "human", n_window: int = 1):

        if not hasattr(self, "artists"):
            self.render_init(n_window)
        state = self._state
        r_rob = self.robot.robot_params["radius"]
        r_obs = self.obses[0].robot_params["radius"]

        def arrow_pos(state):
            x, y, theta = state[0], state[1], state[2]
            return [x, x + np.cos(theta) * r_rob], [y, y + np.sin(theta) * r_rob]

        for i in range(n_window):
            for j in range(n_window):
                idx = i * n_window + j
                circles, arrows = self.artists[idx]
                circles[0].center = state[idx, :2]
                arrows[0].set_data(arrow_pos(state[idx, :5]))
                for k in range(self.n_obstacle):
                    circles[k + 1].center = state[
                        idx, 3 + (k + 1) * 5 : 3 + (k + 1) * 5 + 2
                    ]
                    arrows[k + 1].set_data(
                        arrow_pos(state[idx, 3 + (k + 1) * 5 : 3 + (k + 1) * 5 + 5])
                    )
            plt.pause(0.02)

    def render_init(self, n_window: int = 1):

        fig, axs = plt.subplots(n_window, n_window, figsize=(9, 9))
        artists = []

        r_rob = self.robot.robot_params["radius"]
        r_obs = self.obses[0].robot_params["radius"]
        for i in range(n_window):
            for j in range(n_window):
                if n_window == 1:
                    ax = axs
                else:
                    ax = axs[i, j]
                ax.set_aspect(1)
                ax.set_ylim(-3, 3)
                x = np.linspace(0, 6, 1000)
                y = np.sin(1 / 30 * x)
                ax.plot(x, y, "k")
                circles = []
                arrows = []
                circles.append(plt.Circle([0, 0], r_rob, color="red", fill=False))
                arrows.append(ax.plot([], [], "red")[0])
                ax.add_artist(circles[-1])
                ax.add_artist(arrows[-1])
                for k in range(self.n_obstacle):
                    circles.append(plt.Circle([0, 0], r_obs, color="blue", fill=False))
                    ax.add_artist(circles[-1])

                    arrows.append(ax.plot([], [], "blue")[0])
                artists.append([circles, arrows])
        self.artists = artists
        plt.ion()

    def close(self):
        plt.close("all")


class Robot:
    def __init__(self):
        self.robot_params = dict(
            v_max=0.4,
            w_max=np.pi / 2,
            v_delta_max=1.8,
            w_delta_max=0.8,
            v_desired=0.3,
            radius=0.74 / 2,  # per second
        )
        self.path = ReferencePath()

    def f_xu(
        self, states: np.ndarray, actions: np.ndarray, T: float, type: str
    ) -> np.ndarray:
        v_delta_max = self.robot_params["v_delta_max"]
        v_max = self.robot_params["v_max"]
        w_max = self.robot_params["w_max"]
        w_delta_max = self.robot_params["w_delta_max"]
        std_type = {
            "ego": [0.0, 0.0],
            "obs": [0.03, 0.02],
            "none": [0, 0],
            "explore": [0.3, 0.3],
        }
        stds = std_type[type]

        x, y, theta, v, w = (
            states[:, 0],
            states[:, 1],
            states[:, 2],
            states[:, 3],
            states[:, 4],
        )
        v_cmd, w_cmd = actions[:, 0], actions[:, 1]

        delta_v = np.clip(v_cmd - v, -v_delta_max * T, v_delta_max * T)
        delta_w = np.clip(w_cmd - w, -w_delta_max * T, w_delta_max * T)
        v_cmd = (
            np.clip(v + delta_v, -v_max, v_max)
            + np.random.normal(0, stds[0], [states.shape[0]]) * 0.5
        )
        w_cmd = (
            np.clip(w + delta_w, -w_max, w_max)
            + np.random.normal(0, stds[1], [states.shape[0]]) * 0.5
        )
        next_state = [
            x + T * np.cos(theta) * v_cmd,
            y + T * np.sin(theta) * v_cmd,
            np.clip(theta + T * w_cmd, -np.pi, np.pi),
            v_cmd,
            w_cmd,
        ]

        return np.stack(next_state, 1)

    def tracking_error(self, x: np.ndarray) -> np.ndarray:
        error_position = x[:, 1] - self.path.compute_path_y(x[:, 0])
        error_head = x[:, 2] - self.path.compute_path_phi(x[:, 0])

        error_v = x[:, 3] - self.robot_params["v_desired"]
        tracking = np.concatenate(
            (
                error_position.reshape(-1, 1),
                error_head.reshape(-1, 1),
                error_v.reshape(-1, 1),
            ),
            1,
        )
        return tracking


class ReferencePath(object):
    def __init__(self):
        pass

    def compute_path_y(self, x: np.ndarray) -> np.ndarray:
        y = 0 * np.sin(1 / 3 * x)
        return y

    def compute_path_phi(self, x: np.ndarray) -> np.ndarray:
        deriv = 0 * np.cos(1 / 3 * x)
        return np.arctan(deriv)


def env_creator(**kwargs: Any):
    """
    make env `pyth_mobilerobot`
    """
    return PythMobilerobot(**kwargs)
