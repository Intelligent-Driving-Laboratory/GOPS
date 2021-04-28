cartpole_SAMPLE_TIME = 0.01
import math
from gym import spaces
import gym
import numpy as np
from modules.env.resources import cartpole

class SimuCartpole(gym.Env):

    def __init__(self):
        self._physics = cartpole.cartpoleModelClass_wrapper()
        high = np.array([
            self._physics.get_param()['x_threshold'] * 2, np.finfo(np.float32).max,
            self._physics.get_param()['theta_threshold_radians'] * 2, np.finfo(np.float32).max])
        self.action_space = spaces.Box(low=self._physics.get_param()['f_threshold'][0], high=self._physics.get_param()['f_threshold'][1], shape=(1,))
        self.observation_space = spaces.Box(-high, high)
        self.reset()

    def step(self, action):
        action = { 'Force': action[0]}
        state, is_done, reward = self._step_physics(action)
        self.state = state
        return state, reward, is_done, {}

    def reset(self):
        self._physics.terminate()
        self._physics = cartpole.cartpoleModelClass_wrapper()
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        param = self._physics.get_param()
        param.update(list(zip(('x_o','xdot_o','theta_o','thetadot_o'), self.state.tolist())))
        self._physics.set_param(param)
        self._physics.initialize()
        return self.state

    def render(self):
        pass

    def close(self):
        self._physics.renderterminate()

    def _step_physics(self, action):
        return self._physics.step(action)


if __name__ == "__main__":
    import gym
    import numpy as np

    env = SimuCartpole()
    s = env.reset()
    for i in range(50):
        a = np.ones([1, 1])*2
        sp, r, d, _ = env.step(a)
        print(s, a, r, d)
        s = sp
