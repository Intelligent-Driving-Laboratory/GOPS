#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab


import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym.utils import seeding

from gops.env.resources.car_following_2d.car_following_2d import CarFollowingDynamics2D

gym.logger.setLevel(gym.logger.ERROR)

Y_NAME = ["devi_v", "gap", "action"]
X_RANGE = [(0, 105), (0, 105), (0, 105)]
Y_RANGE = [(-10, +10), (0, 10), (-4, 3)]
COLOR = ["b", "r", "y", "g"]


class PythCarfollowingData:
    def __init__(self, **kwargs):
        self.dyn = CarFollowingDynamics2D()
        self.mode = "training"
        self.constraint_dim = 1
        self.use_constraint = kwargs.get("use_constraint", True)

        lb_state = np.array([-np.inf, -np.inf])
        hb_state = -lb_state
        lb_action = np.array(
            [
                -4.0,
            ]
        )
        hb_action = np.array(
            [
                3.0,
            ]
        )

        self.action_space = spaces.Box(low=lb_action, high=hb_action, dtype=np.float32)
        self.observation_space = spaces.Box(lb_state, hb_state, dtype=np.float32)

        self.seed()

        self.first_rendering = True
        self.state_buffer = np.zeros((20100, 2), dtype=np.float32)
        self.action_buffer = np.zeros((20100,), dtype=np.float32)
        self.steps = 0
        # render variable initial
        self.fig = None
        self.lines = None
        self.axes = None
        self.backgrounds = None
        self.state = self.reset()

    def set_mode(self, mode):
        self.mode = mode

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.action_buffer[self.steps] = action
        state_next = self.dyn.prediction(self.state, action)
        self.state = state_next

        reward = self.dyn.compute_reward(state_next, action)
        ############################################################################################
        # define the constraint here

        ################################################################################################################
        # define the ending condition here the format is just like isdone = l(next_state)

        isdone = bool(state_next[1] < 0.5)

        constraint = self.dyn.compute_cost(state_next, action)
        self.steps += 1
        self.state_buffer[self.steps, :] = self.state
        info = {"TimeLimit.truncated": self.steps > 100, "constraint": constraint}
        if state_next[1] < 2.0 - 0.01:
            info.update({"constraint_violate": True})
        return self.state, reward, isdone, info

    def reset(self):
        self.steps = 0

        devi_v = self.np_random.uniform(-4, 4)
        gap = self.np_random.uniform(5, 10)
        self.state = np.array([devi_v, gap], dtype=np.float32)
        self.state_buffer[self.steps, :] = self.state

        return self.state

    def render(self, mode="human"):
        tt = range(self.steps + 1)
        TT = [tt, tt, tt[0 : self.steps]]
        XX = [
            self.state_buffer[0 : self.steps + 1, 0],
            self.state_buffer[0 : self.steps + 1, 1],
            self.action_buffer[0 : self.steps],
        ]
        if self.first_rendering:
            self.fig, self.axes = plt.subplots(2, 2)
            self.lines = []
            self.axes = self.axes.reshape(-1)
            for idx, ax in enumerate(self.axes):
                if idx == 3:
                    break
                (line,) = ax.plot(TT[idx], XX[idx], color=COLOR[idx])
                self.lines.append(line)
                ax.set_title(Y_NAME[idx])
                ax.set_xlim(X_RANGE[idx])
                ax.set_ylim(Y_RANGE[idx])
                if idx == 1:
                    ax.axhline(y=2.0, ls=":", c="k")
                    ax.axhline(y=2.5, ls=":", c="g")
                    ax.axhline(y=3.0, ls=":", c="r")

            plt.tight_layout()

            if mode == "human":
                plt.show(block=False)

            self.fig.canvas.draw()

            self.first_rendering = False
            self.backgrounds = [self.fig.canvas.copy_from_bbox(ax.bbox) for ax in self.axes]
            # time.sleep(10)

        items = zip(self.axes, self.lines, self.backgrounds)

        for idx, (ax, line, bg) in enumerate(items):
            if idx == 3:
                break
            self.fig.canvas.restore_region(bg)
            line.set_xdata(TT[idx])
            line.set_ydata(XX[idx])
            # self.axes[0].draw_artist(self.axes[0].patch)
            ax.draw_artist(line)
            self.fig.canvas.blit(ax.bbox)

        self.fig.canvas.update()
        self.fig.canvas.flush_events()
        if mode == "rgb_array":
            image_from_plot = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            plt.pause(0.01)
            return image_from_plot

    def close(self):
        pass


def env_creator(**kwargs):
    return PythCarfollowingData()


if __name__ == "__main__":
    # import cv2
    import time

    env = env_creator()

    s = env.reset()
    env.observation_space.contains(s)
    a = env.action_space.sample()
    s, r, d, _ = env.step(a)

    print(type(d))
    # for i in range(100):
    #     a = env.action_space.sample()
    #     x, r, d, info = env.step(a)
    #     # pprint([x, r, d, info])
    #     env.render()

    # fig_array = env.render(mode="human")
    # print(fig_array.shape)
    # cv2.imshow("pic", fig_array)
    # cv2.waitKey(0)
    # cv2.destroyWindow()
    # time.sleep(100)
