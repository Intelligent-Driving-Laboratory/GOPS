import math
import warnings
from typing import Union

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym.spaces import Box
from gym.utils import seeding

warnings.filterwarnings("ignore")
gym.logger.setLevel(gym.logger.ERROR)
MAX_BUFFER = 20100


class LQDynamics:
    def __init__(self, config):

        self.A = torch.as_tensor(config["A"], dtype=torch.float32)
        self.B = torch.as_tensor(config["B"], dtype=torch.float32)
        self.Q = torch.as_tensor(config["Q"], dtype=torch.float32)  # diag vector
        self.R = torch.as_tensor(config["R"], dtype=torch.float32)  # diag vector

        self.time_step = config["dt"]

    def f_xu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
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
        """
        environment dynamics in torch, batch operation,
                        or in numpy, non-batch

        Parameters
        ----------
        x_t: [b,3]
        u_t: [b,1]

        Returns
        -------
        x_next: [b,3]
        """

        numpy_flag = False
        if isinstance(x_t, np.ndarray):
            x_t = torch.as_tensor(x_t, dtype=torch.float32).unsqueeze(0)
            u_t = torch.as_tensor(u_t, dtype=torch.float32).unsqueeze(0)
            numpy_flag = True
        f_dot = self.f_xu(x_t, u_t)
        x_next = f_dot * self.time_step + x_t

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
            x_t = torch.as_tensor(x_t, dtype=torch.float32).unsqueeze(0)
            u_t = torch.as_tensor(u_t, dtype=torch.float32).unsqueeze(0)
            numpy_flag = True
        reward_state = torch.sum(torch.pow(x_t, 2) * self.Q, dim=-1)
        reward_action = torch.sum(torch.pow(u_t, 2) * self.R, dim=-1)
        reward = reward_state + reward_action
        if numpy_flag:
            reward = reward[0].item()
        return reward


class LqEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(self, config, **kwargs):
        self.is_adversary = kwargs.get("is_adversary", False)
        self.is_constraint = kwargs.get("is_constraint", False)

        self.config = config
        self.dynamics = LQDynamics(config)

        state_high = np.array(config["state_high"], dtype=np.float32)
        state_low = np.array(config["state_low"], dtype=np.float32)
        self.observation_space = Box(low=state_low, high=state_high)
        self.observation_dim = self.observation_space.shape[0]

        action_high = np.array(config["action_high"], dtype=np.float32)
        action_low = np.array(config["action_low"], dtype=np.float32)
        self.action_space = Box(low=action_low, high=action_high)
        self.action_dim = self.action_space.shape[0]

        self.seed()

        # environment variable
        self.obs = None

        self.first_rendering = True
        self.state_buffer = np.zeros((MAX_BUFFER, self.observation_dim), dtype=np.float32)
        self.action_buffer = np.zeros((MAX_BUFFER, self.action_dim), dtype=np.float32)
        self.step_counter = 0
        self.num_figures = self.observation_dim + self.action_dim
        self.ncol = math.ceil(math.sqrt(self.num_figures))
        self.nrow = math.ceil(self.num_figures / self.ncol)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, init_state=None):
        self.step_counter = 0

        if init_state is None:
            init_mean = np.array(self.config["init_mean"])
            init_std = np.array(self.config["init_std"])
            self.obs = self.np_random.randn() * init_std + init_mean

        else:
            self.obs = init_state

        self.state_buffer[self.step_counter, :] = self.obs

        return self.obs

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
        self.obs = self.dynamics.prediction(self.obs, action)
        reward = self.dynamics.compute_reward(self.obs, action)

        done = False  # TODO
        info = {}
        self.step_counter += 1
        self.state_buffer[self.step_counter, :] = self.obs

        return self.obs, reward, done, info

    def render(self, mode="human_mode"):
        """
        render 很快
        """
        x_state = range(self.step_counter + 1)
        y_state = self.state_buffer[0: self.step_counter + 1, :]
        x_actions = range(self.step_counter)
        y_action = self.action_buffer[0: self.step_counter, :]

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
                ax = self.fig.add_subplot(self.nrow, self.ncol, idx_a + self.observation_dim + 1)
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
            # time.sleep(10)

        else:
            for idx_s in range(self.observation_dim):
                self.lines[idx_s].set_xdata(x_state)
                self.lines[idx_s].set_ydata(y_state[:, idx_s])
                self.axes[idx_s].draw_artist(self.axes[idx_s].patch)
                self.axes[idx_s].draw_artist(self.lines[idx_s])

            for idx_a in range(self.action_dim):
                self.lines[self.observation_dim + idx_a].set_xdata(x_state)
                self.lines[self.observation_dim + idx_a].set_ydata(y_state[:, idx_a])
                self.axes[self.observation_dim + idx_a].draw_artist(self.axes[self.observation_dim + idx_a].patch)
                self.axes[self.observation_dim + idx_a].draw_artist(self.lines[self.observation_dim + idx_a])

        self.update_canvas()

        self.fig.canvas.flush_events()

    def update_canvas(self):
        if hasattr(self.fig.canvas, "update"):
            self.fig.canvas.update()
        elif hasattr(self.fig.canvas, "draw"):
            self.fig.canvas.draw()
        else:
            raise RuntimeError("In current matplotlib backend, canvas has no attr update or draw, cannot rend")

    def close(self):
        plt.cla()
        plt.clf()


class LqModel(torch.nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        """
        you need to define parameters here
        """
        # define your custom parameters here
        self.dynamics = LQDynamics(config)
        # define common parameters here
        self.state_dim = len(config["state_high"])
        self.action_dim = len(config["action_high"])
        lb_state = np.array(config["state_low"])
        hb_state = np.array(config["state_high"])
        lb_action = np.array(config["action_low"])
        hb_action = np.array(config["action_high"])
        self.dt = config["dt"]  # seconds between state updates

        # do not change the following section
        self.lb_state = torch.tensor(lb_state, dtype=torch.float32)
        self.hb_state = torch.tensor(hb_state, dtype=torch.float32)
        self.lb_action = torch.tensor(lb_action, dtype=torch.float32)
        self.hb_action = torch.tensor(hb_action, dtype=torch.float32)

    def forward(self, state: torch.Tensor, action: torch.Tensor, beyond_done=None):
        """
        rollout the model one step, notice this method will not change the value of self.state
        you need to define your own state transition  function here
        notice that all the variables contains the batch dim you need to remember this point
        when constructing your function
        :param state: datatype:torch.Tensor, shape:[batch_size, state_dim]
        :param action: datatype:torch.Tensor, shape:[batch_size, action_dim]
        :param beyond_done: flag indicate the state is already done which means it will not be calculated by the model
        :return:
                next_state:  datatype:torch.Tensor, shape:[batch_size, state_dim]
                              the state will not change anymore when the corresponding flag done is set to True
                reward:  datatype:torch.Tensor, shape:[batch_size,]
                isdone:   datatype:torch.Tensor, shape:[batch_size,]
                         flag done will be set to true when the model reaches the max_iteration or the next state
                         satisfies ending condition

                info: datatype: dict, any useful information for debug or training, including constraint
                        {"constraint": None}
        """
        warning_msg = "action out of action space!"
        if not ((action <= self.hb_action).all() and (action >= self.lb_action).all()):
            warnings.warn(warning_msg)
            action = clip_by_tensor(action, self.lb_action, self.hb_action)

        warning_msg = "state out of state space!"
        if not ((state <= self.hb_state).all() and (state >= self.lb_state).all()):
            warnings.warn(warning_msg)
            state = clip_by_tensor(state, self.lb_state, self.hb_state)

        #  define your forward function here: the format is just like: state_next = f(state,action)
        state_next = self.dynamics.prediction(state, action)

        ############################################################################################

        # define the ending condation here the format is just like isdone = l(next_state)
        isdone = torch.full([state.size()[0]], False, dtype=torch.bool)

        ############################################################################################

        # define the reward function here the format is just like: reward = l(state,state_next,reward)
        reward = self.dynamics.compute_reward(state, action).reshape(-1)

        ############################################################################################
        if beyond_done is None:
            beyond_done = torch.full([state.size()[0]], False, dtype=torch.float32)

        beyond_done = beyond_done.bool()
        mask = isdone | beyond_done
        mask = torch.unsqueeze(mask, -1)
        state_next = ~mask * state_next + mask * state
        reward = ~(beyond_done) * reward
        return state_next, reward, mask.squeeze(), {"constraint": None}

    def forward_n_step(self, func, n, state: torch.Tensor):
        pass


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min) * t + (t < t_min) * t_min
    result = (result <= t_max) * result + (result > t_max) * t_max
    return result


def test_check():
    from gops.env.resources.linear_quadratic_problem.lq_configs import config_s3a1
    from gops.env.tools.env_check import check_env0
    from gops.env.tools.model_check import check_model0
    env = LqEnv(config_s3a1)
    model = LqModel(config_s3a1)
    check_env0(env)
    check_model0(env, model)


def test_env():
    from gops.env.resources.linear_quadratic_problem.lq_configs import config_s3a1

    env = LqEnv(config_s3a1)
    env.reset()
    for _ in range(100):
        a = env.action_space.sample()
        env.step(a)
        env.render()


if __name__ == "__main__":
    # test_dynamic()
    test_check()
    test_env()
