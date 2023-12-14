#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Linear Quadratic control environment base
#  Update Date: 2022-08-12, Yuhang Zhang: create environment base
#  Update Date: 2022-10-24, Yujie Yang: add wrapper


import math
import warnings
from typing import Tuple, Union

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym.spaces import Box
from scipy.linalg._solvers import solve_discrete_are

from gops.env.env_ocp.pyth_base_env import PythBaseEnv
from gops.env.env_ocp.env_model.pyth_base_model import PythBaseModel
from gops.utils.gops_typing import InfoDict
from gops.env.env_ocp.resources.lq_configs import config_s3a1

warnings.filterwarnings("ignore")
gym.logger.setLevel(gym.logger.ERROR)
MAX_BUFFER = 20100


class LQDynamics:
    def __init__(
        self, config: dict, device: Union[torch.device, str, None] = None,
    ):
        self.A = torch.as_tensor(config["A"], dtype=torch.float32, device=device)
        self.B = torch.as_tensor(config["B"], dtype=torch.float32, device=device)
        self.Q = torch.as_tensor(
            config["Q"], dtype=torch.float32, device=device
        )  # diag vector
        self.R = torch.as_tensor(
            config["R"], dtype=torch.float32, device=device
        )  # diag vector

        self.time_step = config["dt"]
        self.K, self.P = self.compute_control_matrix()

        self.reward_scale = config["reward_scale"]
        self.reward_shift = config["reward_shift"]
        self.state_dim = self.A.shape[0]

        # IA = (1 - A * dt)
        IA = torch.eye(self.state_dim, device=device) - self.A * self.time_step
        self.inv_IA = torch.linalg.pinv(IA)

        self.device = device

    def compute_control_matrix(self):
        gamma = 0.99
        A0 = self.A.cpu().numpy().astype("float64")
        A = np.linalg.pinv(np.eye(A0.shape[0]) - A0 * self.time_step) * np.sqrt(gamma)
        B0 = self.B.cpu().numpy().astype("float64")
        B = A @ B0 * self.time_step
        Q = np.diag(self.Q.cpu().numpy()).astype("float64")
        R = np.diag(self.R.cpu().numpy()).astype("float64")
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A
        return K, P

    def f_xu_old(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        x' = f(x, u)

        Parameters
        ----------
        x: [b,3]
        u: [b,1]

        Returns
        -------
        x_dot: [b,3]
        """
        f_dot = torch.mm(self.A, x.T) + torch.mm(self.B, u.T)
        return f_dot.T

    def prediction(
        self, x_t: Union[torch.Tensor, np.ndarray], u_t: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        numpy_flag = False
        if isinstance(x_t, np.ndarray):
            x_t = torch.as_tensor(
                x_t, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            u_t = torch.as_tensor(
                u_t, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            numpy_flag = True

        tmp = torch.mm(self.B, u_t.T) * self.time_step + x_t.T

        x_next = torch.mm(self.inv_IA, tmp).T

        if numpy_flag:
            x_next = x_next.detach().numpy().squeeze(0)
        return x_next

    def compute_reward(
        self, x_t: Union[torch.Tensor, np.ndarray], u_t: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        reward in torch, batch operation

        Parameters
        ----------
        x_t: [b,3]
        u_t: [b,1]

        Returns
        -------
        reward : [b,]
        """
        numpy_flag = False
        if isinstance(x_t, np.ndarray):
            x_t = torch.as_tensor(
                x_t, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            u_t = torch.as_tensor(
                u_t, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            numpy_flag = True
        reward_state = torch.sum(torch.pow(x_t, 2) * self.Q, dim=-1)
        reward_action = torch.sum(torch.pow(u_t, 2) * self.R, dim=-1)
        reward = self.reward_scale * (
            self.reward_shift - 1.0 * (reward_state + reward_action)
        )
        if numpy_flag:
            reward = reward[0].item()
        return reward


class LqEnv(PythBaseEnv):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(self, config=config_s3a1, **kwargs):
        work_space = kwargs.pop("work_space", None)
        if work_space is None:
            init_mean = np.array(config["init_mean"], dtype=np.float32)
            init_std = np.array(config["init_std"], dtype=np.float32)
            work_space = np.stack((init_mean - 3 * init_std, init_mean + 3 * init_std))
        super(LqEnv, self).__init__(work_space=work_space, **kwargs)

        self.is_adversary = kwargs.get("is_adversary", False)
        self.is_constraint = kwargs.get("is_constraint", False)

        self.config = config
        self.max_episode_steps = config["max_step"]
        self.dynamics = LQDynamics(config)

        state_high = np.array(config["state_high"], dtype=np.float32)
        state_low = np.array(config["state_low"], dtype=np.float32)
        self.observation_space = Box(low=state_low, high=state_high)
        self.observation_dim = self.observation_space.shape[0]

        action_high = np.array(config["action_high"], dtype=np.float32)
        action_low = np.array(config["action_low"], dtype=np.float32)
        self.action_space = Box(low=action_low, high=action_high)
        self.action_dim = self.action_space.shape[0]
        self.control_matrix = self.dynamics.K

        self.seed()

        # environment variable
        self.obs = None

        self.first_rendering = True
        self.state_buffer = np.zeros(
            (MAX_BUFFER, self.observation_dim), dtype=np.float32
        )
        self.action_buffer = np.zeros((MAX_BUFFER, self.action_dim), dtype=np.float32)
        self.step_counter = 0
        self.num_figures = self.observation_dim + self.action_dim
        self.ncol = math.ceil(math.sqrt(self.num_figures))
        self.nrow = math.ceil(self.num_figures / self.ncol)

    @property
    def has_optimal_controller(self):
        return True

    def control_policy(self, state, info):
        return -self.control_matrix @ state

    def reset(self, init_state=None, **kwargs):
        self.step_counter = 0

        if init_state is None:
            self.obs = self.sample_initial_state()
        else:
            self.obs = np.array(init_state, dtype=np.float32)

        self.state_buffer[self.step_counter, :] = self.obs

        return self.obs, {}

    def step(self, action: np.ndarray, adv_action=None):
        """
        action: datatype:numpy.ndarray, shape:[action_dim,]
        adv_action: datatype:numpy.ndarray, shape:[adv_action_dim,]
        return:
        self.obs: next observation, datatype:numpy.ndarray, shape:[state_dim]
        reward: reward signal
        done: done signal, datatype: bool
        """

        # define environment transition, reward,  done signal  and constraint function here

        self.action_buffer[self.step_counter] = action
        reward = self.dynamics.compute_reward(self.obs, action)
        self.obs = self.dynamics.prediction(self.obs, action)

        done = self.is_done(self.obs)
        if done:
            reward -= 100
        info = {}
        self.step_counter += 1
        self.state_buffer[self.step_counter, :] = self.obs
        return self.obs, reward, done, info

    def is_done(self, obs):
        high = self.observation_space.high
        low = self.observation_space.low
        return bool(np.any(obs > high) or np.any(obs < low))

    def render(self, mode="human_mode"):
        """
        render 很快
        """
        x_state = range(self.step_counter + 1)
        y_state = self.state_buffer[0 : self.step_counter + 1, :]
        x_actions = range(self.step_counter)
        y_action = self.action_buffer[0 : self.step_counter, :]

        if self.first_rendering:
            self.fig = plt.figure()
            self.lines = []
            self.axes = []

            for idx_s in range(self.observation_dim):
                ax = self.fig.add_subplot(self.nrow, self.ncol, idx_s + 1)
                (line,) = ax.plot(x_state, y_state[:, idx_s], color="b")
                self.lines.append(line)
                self.axes.append(ax)
                ax.set_title(f"state-{idx_s}")
                yiniy = y_state[0, idx_s]
                ax.set_ylim([yiniy - 1, yiniy + 1])
                ax.set_xlim([0, self.config["max_step"] + 5])

            for idx_a in range(self.action_dim):
                ax = self.fig.add_subplot(
                    self.nrow, self.ncol, idx_a + self.observation_dim + 1
                )
                (line,) = ax.plot(x_actions, y_action[:, idx_a], color="b")
                self.lines.append(line)
                self.axes.append(ax)
                ax.set_title(f"action-{idx_a}")
                yiniy = y_action[0, idx_a]
                ax.set_ylim([yiniy - 1, yiniy + 1])
                ax.set_xlim([0, self.config["max_step"] + 5])

            plt.tight_layout()
            plt.show(block=False)

            self.update_canvas()

            self.first_rendering = False

        else:
            for idx_s in range(self.observation_dim):
                self.lines[idx_s].set_xdata(x_state)
                self.lines[idx_s].set_ydata(y_state[:, idx_s])
                self.axes[idx_s].draw_artist(self.axes[idx_s].patch)
                self.axes[idx_s].draw_artist(self.lines[idx_s])

            for idx_a in range(self.action_dim):
                self.lines[self.observation_dim + idx_a].set_xdata(x_state)
                self.lines[self.observation_dim + idx_a].set_ydata(y_state[:, idx_a])
                self.axes[self.observation_dim + idx_a].draw_artist(
                    self.axes[self.observation_dim + idx_a].patch
                )
                self.axes[self.observation_dim + idx_a].draw_artist(
                    self.lines[self.observation_dim + idx_a]
                )

        self.update_canvas()

        self.fig.canvas.flush_events()

    def update_canvas(self):
        if hasattr(self.fig.canvas, "update"):
            self.fig.canvas.update()
        elif hasattr(self.fig.canvas, "draw"):
            self.fig.canvas.draw()
        else:
            raise RuntimeError(
                "In current matplotlib backend, canvas has no attr update or draw, cannot rend"
            )

    def close(self):
        plt.cla()
        plt.clf()


class LqModel(PythBaseModel):
    def __init__(
        self, config: dict, device: Union[torch.device, str, None] = None,
    ):
        """
        you need to define parameters here
        """
        lb_state = np.array(config["state_low"])
        hb_state = np.array(config["state_high"])
        lb_action = np.array(config["action_low"])
        hb_action = np.array(config["action_high"])
        super().__init__(
            obs_dim=lb_state.shape[0],
            action_dim=lb_action.shape[0],
            dt=config["dt"],
            obs_lower_bound=lb_state,
            obs_upper_bound=hb_state,
            action_lower_bound=lb_action,
            action_upper_bound=hb_action,
            device=device,
        )

        # define your custom parameters here
        self.dynamics = LQDynamics(config, device)
        self.P = torch.as_tensor(self.dynamics.P, dtype=torch.float32)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        done: torch.Tensor,
        info: InfoDict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        next_obs = self.dynamics.prediction(obs, action)
        reward = self.dynamics.compute_reward(obs, action).reshape(-1)
        done = torch.full([obs.size()[0]], False, dtype=torch.bool, device=self.device)
        info = {"constraint": None}
        return next_obs, reward, done, info

    def get_terminal_cost(self, obs: torch.Tensor) -> torch.Tensor:
        return obs @ self.P @ obs.T


def test_check():
    from gops.env.inspector.env_data_checker import check_env0
    from gops.env.inspector.env_model_checker import check_model0

    env = LqEnv(config_s3a1)
    model = LqModel(config_s3a1)
    check_env0(env)
    check_model0(env, model)


def test_env():
    from gops.env.env_ocp.resources.lq_configs import config_s5a1

    env = LqEnv(config_s5a1)
    env.reset()
    print(env.has_optimal_controller)
    print(env.control_matrix)
    a0 = env.control_policy(np.array([0.1, 0.2, -0.3, 0, 0]))
    print(a0)
    for _ in range(100):
        a = env.action_space.sample()
        env.step(a)
        env.render()


if __name__ == "__main__":
    # test_dynamic()
    # test_check()
    test_env()
