#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces
from gym.wrappers.time_limit import TimeLimit

from gops.env.env_ocp.pyth_base_data import PythBaseEnv
from gops.env.env_ocp.pyth_idpendulum_model import Dynamics

gym.logger.setLevel(gym.logger.ERROR)
plt.rcParams['toolbar'] = 'None'


class PythInverteddoublependulum(PythBaseEnv):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(self, **kwargs):
        work_space = kwargs.pop("work_space", None)
        if work_space is None:
            # initial range of [p, theta1, theta2, pdot, theta1dot, theta2dot]
            init_high = np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3], dtype=np.float32)
            init_low = -init_high
            work_space = np.stack((init_low, init_high))
        super(PythInverteddoublependulum, self).__init__(work_space=work_space, **kwargs)

        self.is_adversary = kwargs.get("is_adversary", False)
        self.is_constraint = kwargs.get("is_constraint", False)

        # define your custom parameters here

        self.dynamics = Dynamics()
        self.tau = 0.05
        # define observation space here
        hb_observation = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
        self.observation_space = spaces.Box(
            low=-np.array(hb_observation, dtype=np.float32), high=np.array(hb_observation, dtype=np.float32)
        )

        # define action space here
        lb_action = [-1.0]
        hb_action = [1.0]
        self.action_space = spaces.Box(
            low=np.array(lb_action, dtype=np.float32), high=np.array(hb_action, dtype=np.float32)
        )

        self.seed()
        plt.ion()
        self.obs = None

    def step(self, action: np.ndarray, adv_action=None):
        """
        action: datatype:numpy.ndarray, shape:[action_dim,]
        adv_action: datatype:numpy.ndarray, shape:[adv_action_dim,]
        return:
        self.obs: next observation, datatype:numpy.ndarray, shape:[state_dim]
        reward: reward signal
        done: done signal, datatype: bool
        """
        action = 500.0 * action
        # define environment transition, reward,  done signal  and constraint function here
        obs_batch = torch.as_tensor(self.obs, dtype=torch.float32).reshape(1, -1)
        act_batch = torch.as_tensor(action, dtype=torch.float32).reshape(1, -1)
        next_obs_batch = self.dynamics.f_xu(obs_batch, act_batch, self.tau)
        reward = self.dynamics.compute_rewards(next_obs_batch)
        done = self.dynamics.get_done(next_obs_batch)
        info = {}

        self.obs = next_obs_batch.numpy()[0]
        reward = reward.numpy()[0]
        return self.obs, reward, bool(done), info

    def reset(self, *, init_obs=None,**kwargs):
        """
        self.obs: initial observation, datatype:numpy.ndarray, shape:[state_dim]
        """
        # define initial state distribution here
        if init_obs is None:
            self.obs = self.sample_initial_state()
        else:
            assert self.observation_space.contains(init_obs)
            self.obs = init_obs
        return self.obs

    def render(self, mode="human"):
        plt.cla()
        states = self.obs
        p, theta1, theta2, pdot, theta1dot, theta2dot = (
            states[0],
            states[1],
            states[2],
            states[3],
            states[4],
            states[5],
        )
        point0x, point0y = p, 0
        point1x, point1y = point0x + self.dynamics.l_rod1 * np.sin(
            theta1
        ), point0y + self.dynamics.l_rod1 * np.cos(theta1)
        point2x, point2y = point1x + self.dynamics.l_rod2 * np.sin(
            theta2
        ), point1y + self.dynamics.l_rod2 * np.cos(theta2)

        plt.title("Inverted Double Pendulum")
        ax = plt.gca() # plt.axes(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
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
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.pause(0.01)
            return image_from_plot
        elif mode == "human":
            plt.pause(0.01)
            plt.show()
    def close(self):
        plt.cla()
        plt.clf()


def env_creator(**kwargs):
    """
    make env `pyth_inverteddoublependulum`
    """
    return TimeLimit(PythInverteddoublependulum(**kwargs), 1000)


if __name__ == "__main__":
    env = env_creator()
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        s, r, d, _ = env.step(action)
        print(s)
        env.render()
        if d: env.reset()
