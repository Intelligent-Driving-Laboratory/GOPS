from typing import Tuple, Optional, Sequence

import gym
from gym import spaces
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from gops.env.env_gen_ocp.context.balance_point import BalancePoint
from gops.env.env_gen_ocp.pyth_base import Env, State
from gops.env.env_gen_ocp.robot.Idpendulum_dynamics import Dynamics


matplotlib.use("Agg")
gym.logger.setLevel(gym.logger.ERROR)
plt.rcParams["toolbar"] = "None"

class Inverteddoublependulum(Env):
    def __init__(self, **kwargs):
        self.robot: Dynamics = Dynamics()
        self.context: BalancePoint = BalancePoint(
            balanced_state=np.array([0, 0, 0, 0, 0, 0]),
            index=[0, 1, 2],
        )
        obs_dim = 6
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * obs_dim),
            high=np.array([np.inf] * obs_dim),
        )
        self.action_space = self.robot.action_space
        self.max_episode_steps = 500
        self.seed()

    def reset(
        self, 
        seed: Optional[int] = None, 
        init_state: Optional[Sequence] = None
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.seed(seed)
        if init_state is None:
            high = np.array([5, 0.1, 0.1, 0.3, 0.3, 0.3], dtype=np.float32)
            init_state = self.np_random.uniform(low=-high, high=high).astype(np.float32)
        self._state = State(
            robot_state=self.robot.reset(init_state),
            context_state=self.context.reset(),
        )
        return self._get_obs(), self._get_info()

    def _get_obs(self) -> np.ndarray:
        return self.robot.state

    def _get_reward(self, action: np.ndarray) -> float:
        action = action.squeeze(-1)
        balanced_state = np.zeros_like(self.robot.state)
        balanced_state[self.context.index] = self.context.state.reference
        trans_state = self.robot.state - balanced_state
        p, theta1, theta2, pdot, theta1dot, theta2dot = (
            trans_state[0],
            trans_state[1],
            trans_state[2],
            trans_state[3],
            trans_state[4],
            trans_state[5],
        )
        dist_penalty = (
            0 * np.square(p) + 5 * np.square(theta1) + 10 * np.square(theta2)
        )
        v0, v1, v2 = pdot, theta1dot, theta2dot
        vel_penalty = (
            0.5 * np.square(v0) + 0.5 * np.square(v1) + 1 * np.square(v2)
        )
        act_penalty = 1 * np.square(action)
        rewards = 10 - dist_penalty - vel_penalty - act_penalty
        return rewards

    def _get_terminated(self) -> bool:
        balanced_state = np.zeros_like(self.robot.state)
        balanced_state[self.context.index] = self.context.state.reference
        trans_state = self.robot.state - balanced_state
        p, theta1, theta2, pdot, theta1dot, theta2dot = (
            trans_state[0],
            trans_state[1],
            trans_state[2],
            trans_state[3],
            trans_state[4],
            trans_state[5],
        )
        point0x, point0y = p, 0
        l_rod1, l_rod2 = self.robot.param.l_rod1, self.robot.param.l_rod2
        point1x, point1y = (
            point0x + l_rod1 * np.sin(theta1),
            point0y + l_rod1 * np.cos(theta1),
        )
        point2x, point2y = (
            point1x + l_rod2 * np.sin(theta2),
            point1y + l_rod2 * np.cos(theta2),
        )
        d1 = point2y <= 1.0
        d2 = np.abs(point0x) >= 15
        return np.logical_or(d1, d2)


    def render(self, mode="human"):
        plt.cla()
        p, theta1, theta2, pdot, theta1dot, theta2dot = (
            self.robot.state[0],
            self.robot.state[1],
            self.robot.state[2],
            self.robot.state[3],
            self.robot.state[4],
            self.robot.state[5],
        )
        point0x, point0y = p, 0
        l_rod1, l_rod2 = self.robot.param.l_rod1, self.robot.param.l_rod2
        point1x, point1y = (
            point0x + l_rod1 * np.sin(theta1),
            point0y + l_rod1 * np.cos(theta1),
        )
        point2x, point2y = (
            point1x + l_rod2 * np.sin(theta2),
            point1y + l_rod2 * np.cos(theta2),
        )

        plt.title("Inverted Double Pendulum")
        ax = plt.gca()
        fig = plt.gcf()
        ax.set_xlim((-2.5, 2.5))
        ax.set_ylim((-2.5, 2.5))
        ax.add_patch(
            plt.Rectangle((-2.5, -2.5), 5, 5, edgecolor="black", facecolor="none")
        )
        ax.axis("equal")
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.axis("off")
        ax.plot([-2.5, 2.5], [0, 0], "k")
        ax.plot([-1, -1], [-2.5, 2.5], "k")
        ax.plot([1, 1], [-2.5, 2.5], "k")
        ax.plot(point0x, point0y, "b.")
        ax.plot([point0x, point1x], [point0y, point1y], color="b", linewidth=3.0)
        ax.plot(point1x, point1y, "y.")
        ax.plot([point1x, point2x], [point1y, point2y], color="y", linewidth=3.0)
        ax.plot(point2x, point2y, "r.")

        if mode == "rgb_array":
            plt.show()
            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(
                fig.canvas.get_width_height()[::-1] + (3,)
            )
            plt.pause(0.01)
            return image_from_plot
        elif mode == "human":
            plt.pause(0.01)
            plt.show()

    def close(self):
        plt.cla()
        plt.clf()
    

def env_creator(**kwargs) -> Env:
    return Inverteddoublependulum(**kwargs)
