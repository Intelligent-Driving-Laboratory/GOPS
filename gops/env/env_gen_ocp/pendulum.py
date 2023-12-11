from os import path
from typing import Optional, Sequence, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
from gops.env.env_gen_ocp.pyth_base import Env, State
from gops.env.env_gen_ocp.robot.pendulum_dynamics import PendulumDynamics
from gops.env.env_gen_ocp.context.balance_point import BalancePoint
from gops.utils.math_utils import angle_normalize


class Pendulum(Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, **kwargs):
        self.robot = PendulumDynamics()

        self.context = BalancePoint(
            balanced_state=np.array([0., 0.,], dtype=np.float32),
        )

        high = np.array([1.0, 1.0, self.robot.param.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high)
        self.action_space = self.robot.action_space

        self.render_mode = render_mode

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

    def reset(
        self, 
        seed: Optional[int] = None, 
        init_state: Optional[Sequence] = None,
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.seed(seed)

        if init_state is None:
            high = np.array([np.pi, 1.0], dtype=np.float32)
            init_state = self.np_random.uniform(low=-high, high=high).astype(np.float32)

        self._state = State(
            robot_state=self.robot.reset(init_state),
            context_state=self.context.reset(),
        )

        self.last_u = None

        return self._get_obs(), {}

    def _get_obs(self):
        theta, thetadot = self.robot.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def _get_reward(self, action: np.ndarray) -> float:
        th, thdot = self.state.robot_state
        th_targ, thdot_targ = self.context.state.reference
        u = np.clip(action, self.robot.action_space.low, self.robot.action_space.high)[0]
        self.last_u = u
        costs = (angle_normalize(th) - th_targ) ** 2 + \
            0.1 * (thdot - thdot_targ) ** 2 + 0.001 * (u ** 2)
        return -costs

    def _get_terminated(self) -> bool:
        return False

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state.robot_state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state.robot_state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        img = pygame.image.load(fname)
        if self.last_u is not None:
            scale_img = pygame.transform.smoothscale(
                img,
                (scale * np.abs(self.last_u) / 2, scale * np.abs(self.last_u) / 2),
            )
            is_flip = bool(self.last_u > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            self.surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
