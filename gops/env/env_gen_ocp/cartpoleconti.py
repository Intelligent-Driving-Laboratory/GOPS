from typing import Dict, Optional, Sequence, Tuple

from gym.wrappers.time_limit import TimeLimit
import numpy as np

from gops.env.env_gen_ocp.context.balance_point import BalancePoint
from gops.env.env_gen_ocp.pyth_base import Env, State
from gops.env.env_gen_ocp.robot.cartpole_dynamics import Dynamics


class Cartpoleconti(Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, **kwargs):
        self.robot: Dynamics = Dynamics()
        self.context: BalancePoint = BalancePoint(
            balanced_state=np.array([0., 0., 0., 0.], dtype=np.float32),
            index=[0, 2],
        )
        self.observation_space = self.robot.state_space
        self.action_space = self.robot.action_space
        
        self.seed()
        self.viewer = None
        self._state = None
        self.steps_beyond_done = None
        self.done = False

    def reset(
        self, 
        seed: Optional[int] = None, 
        init_state: Optional[Sequence] = None
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.seed(seed)
        if init_state is None:
            init_state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self._state = State(
            robot_state=self.robot.reset(init_state),
            context_state=self.context.reset(),
        )
        self.done = False
        self.steps_beyond_done = None
        return self._get_obs(), self._get_info()
    
    def _get_obs(self) -> np.ndarray:
        return self.robot.state
    
    def _get_reward(self, action: np.ndarray) -> float:
        return 0.0 if self.done else 1.0
    
    def _get_terminated(self) -> bool:
        balanced_state = np.zeros_like(self.robot.state)
        balanced_state[self.context.index] = self.context.state.reference
        x, x_dot, theta, theta_dot = self.robot.state - balanced_state
        self.done = (
            x < -self.robot.x_threshold
            or x > self.robot.x_threshold
            or theta < -self.robot.theta_threshold_radians
            or theta > self.robot.theta_threshold_radians
        )
        return self.done
    
    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        # TOP OF CART
        carty = 100
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        # MIDDLE OF CART
        cartx = x[0] * scale + screen_width / 2.0
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=(mode == "rgb_array"))

    def close(self):
        if self.viewer:
            self.viewer.close()
    

def env_creator(**kwargs):
    return TimeLimit(Cartpoleconti(**kwargs), max_episode_steps=200)