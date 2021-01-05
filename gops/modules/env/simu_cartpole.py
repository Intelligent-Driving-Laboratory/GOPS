cartpole_SAMPLE_TIME = 0.01
import math
from gym import spaces
import gym
import numpy as np
from resources import cartpole

class SimuCartpole(gym.Env):
    reward_range = (-float('inf'), float('inf'))
    def __init__(self):
        self._physics = None

        # define action and state spaces
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.min_action = -1.0
        self.max_action = 1.0
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,)
        )
        self.observation_space = spaces.Box(-high, high)

        self.reset()

    def step(self, action):
        action = { 'Force': action[0,0]}
        state = self._step_physics(action)[0][[0, 1, 3, 4]]

        # reward and is_done is computed in python instead of simulink
        x, x_dot, theta, theta_dot = state
        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        else:
            reward = 0.0
        ret = state, reward, done, {}
        # state = self._step_physics(action)
        # ret = np.array(state, dtype=ExtY_dtype), self.get_reward(state, action), self.is_done(state), {}
        self.state = state
        return ret

    def seed(self, seed=None):
        '''
        NOTE: Ramdomness is not properly handled yet !!!
        '''
        return [seed]

    def reset(self):
        '''Reset the environment.'''

        if self._physics is not None:
            self._physics.terminate()
        self._physics = cartpole.cartpoleModelClass_wrapper()
        self._physics.initialize()
        #self.state = SimuCartpole.initial_state
        self.state, _, _, _ = self.step(np.zeros([1,1])+2*np.random.random()-1)
        return self.state

    def render(self, mode='human'):
        '''Render the environment.'''
        super(SimuCartpole, self).render(mode=mode)  # Just raise an exception

    def close(self):
        self._physics.renderterminate()

    def _step_physics(self, action):
        return self._physics.step(action)


if __name__ == "__main__":
    import gym
    import numpy as np

    env = SimuCartpole()
    s = env.reset()
    for i in range(20):
        a = np.ones([1,1])
        sp, r, d, _ = env.step(a)
        print(s, a, r)
        s = sp
