from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from gym import spaces
from enum import IntEnum
from gops.env.env_gen_ocp.pyth_base import Env, State
from gops.env.env_gen_ocp.robot.quadrotor_3dof import Quadrotor
from gops.env.env_gen_ocp.context.quad_ref_traj import QuadContext

class QuadType(IntEnum):
    '''Quadrotor types numeration class.'''
    ONE_D = 1  # One-dimensional (along z) movement.
    TWO_D = 2  # Two-dimensional (in the x-z plane) movement.
    THREE_D = 3  # Three-dimensional movement.
    
class QuadTracking(Env):
    def __init__(
        self,       
        **kwargs,
    ):
        self.robot: Quadrotor = Quadrotor(
        )
        self.context: QuadContext = QuadContext(
            quad_type = QuadType.THREE_D
        )
        self.state_space = self.robot.state_space
        self.action_space = self.robot.action_space
        self.max_episode_steps = 200
        self.seed()

    def reset(
        self,
    ) -> Tuple[np.ndarray, dict]:
        return self.robot.reset()

    def _get_obs(self) -> np.ndarray:
        return self.robot._get_obs()

    def _get_reward(self, action: np.ndarray) -> float:
        return self.robot._get_reward(action)

    def _get_terminated(self) -> bool:
        return self.robot._get_done()
    
    def _get_info(self) -> dict:
        return self.robot._get_info()

    def render(self, mode="human"):
        pass
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        reward = self._get_reward(action)
        self._state = self._get_next_state(action)
        terminated = self._get_terminated()
        return self._get_obs(), reward, terminated, self._get_info()
    
    def _get_next_state(self, action: np.ndarray) -> State[np.ndarray]:
        return State(
            robot_state=self.robot.step(action),
            context_state=None
        )
        
def env_creator(**kwargs):
    return QuadTracking(**kwargs)


if __name__ == "__main__":
    # test consistency with old environment
    # for quad_type in QuadType:  
    #     import configparser
    #     import os
    #     config = configparser.ConfigParser()
    #     config['quad_type'] = {'quad_type': str(QuadType.ONE_D)}
    #     with open('./config.ini', 'w') as configfile:
    #         config.write(configfile)
    #     print('\n----------quad_type:',quad_type,'----------')
    env_new = QuadTracking()
    seed = 1
    env_new.seed(seed)
    np.random.seed(seed)
    obs_new = env_new.reset()
    print("reset obs close:", obs_new)
    action = np.random.random(4)
    next_obs_new, reward_new, done_new, _ = env_new.step(action)
    print("step reward close:",  reward_new)

    


