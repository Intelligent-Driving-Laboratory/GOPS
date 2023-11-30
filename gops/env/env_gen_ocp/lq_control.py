import math
from matplotlib import pyplot as plt

import numpy as np
from gym.spaces import Box
from gops.env.env_gen_ocp.context import lq_configs
from gops.env.env_gen_ocp.pyth_base import Env, State
from gops.env.env_gen_ocp.robot.lq import LqModel

MAX_BUFFER = 20100


class LqControl(Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(self, config=lq_configs.config_s3a1, **kwargs):
        work_space = kwargs.pop("work_space", None)
        if work_space is None:
            init_mean = np.array(config["init_mean"], dtype=np.float32)
            init_std = np.array(config["init_std"], dtype=np.float32)
            work_space = np.stack((init_mean - 3 * init_std, init_mean + 3 * init_std))

        self.is_adversary = kwargs.get("is_adversary", False)
        self.is_constraint = kwargs.get("is_constraint", False)

        self.config = config
        self.max_episode_steps = config["max_step"]
        self.robot = LqModel(config)
        self.context = lq_configs.LQContext(balanced_state=[0, 0])
        self.work_space = work_space
        self.initial_distribution = "uniform"

        state_high = np.array(config["state_high"], dtype=np.float32)
        state_low = np.array(config["state_low"], dtype=np.float32)
        self.observation_space = Box(low=state_low, high=state_high)
        self.observation_dim = self.observation_space.shape[0]

        action_high = np.array(config["action_high"], dtype=np.float32)
        action_low = np.array(config["action_low"], dtype=np.float32)
        self.action_space = Box(low=action_low, high=action_high)
        self.action_dim = self.action_space.shape[0]
        self.control_matrix = self.robot.K

        self.seed()

        # environment variable
        self.observation = None

        self.first_rendering = True
        self.state_buffer = np.zeros(
            (MAX_BUFFER, self.observation_dim), dtype=np.float32
        )
        self.action_buffer = np.zeros((MAX_BUFFER, self.action_dim), dtype=np.float32)
        self.step_counter = 0
        self.num_figures = self.observation_dim + self.action_dim
        self.ncol = math.ceil(math.sqrt(self.num_figures))
        self.nrow = math.ceil(self.num_figures / self.ncol)

    def reset(self, init_state=None, **kwargs):
        self.step_counter = 0

        if init_state is None:
            self.observation = self.sample_initial_state()
        else:
            self.observation = np.array(init_state, dtype=np.float32)

        self.robot.reset(self.observation)

        self._state = State(
            robot_state=self.robot.reset(self.observation),
            context_state=self.context.reset(),
        )

        return self._get_obs(), self._get_info()

    def _get_obs(self) -> np.ndarray:
        return self.robot.state

    def _get_reward(self, action: np.ndarray) -> float:
        x_t = self.robot.state
        u_t = action
        x_t = np.expand_dims(x_t, axis=0)
        u_t = np.expand_dims(u_t, axis=0)

        reward_state = np.sum(np.power(x_t, 2) * self.robot.Q, axis=-1)
        reward_action = np.sum(np.power(u_t, 2) * self.robot.R, axis=-1)
        reward = self.robot.reward_scale * (
            self.robot.reward_shift - 1.0 * (reward_state + reward_action)
        )
        reward = reward[0].item()

        return reward

    def _get_terminated(self) -> bool:
        obs = self.robot.state
        high = self.observation_space.high
        low = self.observation_space.low
        return bool(np.any(obs > high) or np.any(obs < low))

    def render(self, mode="human_mode"):
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

    def sample_initial_state(self):
        if self.initial_distribution == "uniform":
            state = self.np_random.uniform(
                low=self.work_space[0], high=self.work_space[1]
            )
        elif self.initial_distribution == "normal":
            mean = (self.work_space[0] + self.work_space[1]) / 2
            std = (self.work_space[1] - self.work_space[0]) / 6
            state = self.np_random.normal(loc=mean, scale=std)
        else:
            raise ValueError(
                f"Invalid initial distribution: {self.initial_distribution}!"
            )
        return state


def env_creator(**kwargs):
    """
    Create an LQ environment with the given configuration.
    """
    lqc = kwargs.get("lq_config", None)
    if lqc is None:
        config = lq_configs.config_s3a1
    elif isinstance(lqc, str):
        assert hasattr(lq_configs, "config_" + lqc)
        config = getattr(lq_configs, "config_" + lqc)
    elif isinstance(lqc, dict):
        config = lqc

    else:
        raise RuntimeError("lq_config invalid")
    lq_configs.check_lq_config(config)

    return LqControl(config, **kwargs)


