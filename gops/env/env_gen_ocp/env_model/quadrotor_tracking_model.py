from typing import Dict, Optional, Tuple, Union

import torch
from enum import IntEnum,Enum
from gops.env.env_gen_ocp.env_model.pyth_base_model import EnvModel
from gops.env.env_gen_ocp.pyth_base import State
from gops.env.env_gen_ocp.context.quad_ref_traj import QuadContext
# import gops.env.env_gen_ocp.robot.quadrotor_model as model

class Cost(str, Enum):
    '''Reward/cost functions enumeration class.'''
    RL_REWARD = 'rl_reward'  # Default RL reward function.
    QUADRATIC = 'quadratic'  # Quadratic cost.


class Task(str, Enum):
    '''Environment tasks enumeration class.'''
    STABILIZATION = 'stabilization'  # Stabilization task.
    TRAJ_TRACKING = 'traj_tracking'  # Trajectory tracking task.

class QuadType(IntEnum):
    '''Quadrotor types numeration class.'''
    ONE_D = 1  # One-dimensional (along z) movement.
    TWO_D = 2  # Two-dimensional (in the x-z plane) movement.
    THREE_D = 3  # Three-dimensional movement.
    
    
    
class QuadType(IntEnum):
    '''Quadrotor types numeration class.'''
    ONE_D = 1  # One-dimensional (along z) movement.
    TWO_D = 2  # Two-dimensional (in the x-z plane) movement.
    THREE_D = 3  # Three-dimensional movement.
    
class QuadTracking(EnvModel):
    def __init__(
        self,       
        **kwargs,
    ):
        self.robot: model.QuadrotorDynMdl = model.QuadrotorDynMdl(
            quad_type = QuadType.THREE_D,
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
    ) -> Tuple[torch.tensor, dict]:
        return self.robot.reset()

    def get_obs(self) -> torch.tensor:
        return self.robot._get_obs()

    def get_reward(self, action: torch.tensor) -> float:
        return self.robot._get_reward(action)

    def get_terminated(self) -> bool:
        return self.robot._get_done()


  
    
    
 
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
    env_new = QuadTracking(quad_type = QuadType.THREE_D)
    seed = 1
    env_new.seed(seed)
    torch.random.seed(seed)
    obs_new = env_new.reset()
    print("reset obs close:", obs_new)
    action = torch.random.random(4)
    state = env_new.get_next_state(obs_new,action)
    print("step reward close:",  state)