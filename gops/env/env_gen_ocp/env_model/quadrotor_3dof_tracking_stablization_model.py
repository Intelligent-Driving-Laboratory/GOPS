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
    
class QuadTracking(EnvModel):
    def __init__(
        self,     
        seed,  
        device: Union[torch.device, str, None] = None,

        **kwargs,
    ):
        dt = 0.01
        action_dim=2,
        super().__init__(
            obs_dim= self.robot.state_space,
            dt=dt,
            obs_lower_bound=None,
            obs_upper_bound=None,
            action_lower_bound=-torch.ones(action_dim),
            action_upper_bound= torch.ones(action_dim),
            device = device,
        )
        self.robot: QuadrotorDynMd2 = QuadrotorDynMd2(
            quad_type = QuadType.THREE_D,
        )
        self.context: QuadContext = QuadContext(
            quad_type = QuadType.THREE_D
        )
        self.state_space = self.robot.state_space
        self.action_space = self.robot.action_space
        self.max_episode_steps = 200
        self.seed = seed



    def reset(self) -> Tuple[torch.Tensor, dict]:
        numpy_state = self.robot.reset()
        tensor_state = torch.from_numpy(numpy_state).float()
        self._state = {
            'robot_state': tensor_state,
            'context_state': None  
        }
        return tensor_state, self._state
 

    def get_obs(self) -> torch.Tensor:
        return torch.from_numpy(self.state.robot_state).float() if isinstance(self.state.robot_state, np.ndarray) else self.state.robot_state

    def get_reward(self, action: torch.Tensor) -> float:
        act_error = action - torch.tensor(self.context.U_GOAL)
        if self.task == 'STABILIZATION':
            state_error = torch.tensor(self.state) - torch.tensor(self.context.X_GOAL)
            dist = torch.sum(self.context.rew_state_weight * state_error ** 2)
            dist += torch.sum(self.context.rew_act_weight * act_error ** 2)
        elif self.task == 'TRAJ_TRACKING':
            wp_idx = min(self.robot.ctrl_step_counter + 1, self.context.X_GOAL.shape[0] - 1)
            state_error = torch.tensor(self.state.robot_state) - torch.tensor(self.context.X_GOAL[wp_idx])
            dist = torch.sum(self.context.rew_state_weight * state_error ** 2)
            dist += torch.sum(self.context.rew_act_weight * act_error ** 2)
        rew = -dist.item()
        if self.robot.rew_exponential:
            rew = torch.exp(rew)
        return rew

    def get_terminated(self) -> bool:
        if self.task == 'STABILIZATION':
            self.goal_reached = (torch.linalg.norm(torch.tensor(self.state) - torch.tensor(self.context.X_GOAL)) < self.context.TASK_INFO['stabilization_goal_tolerance']).item()
            if self.goal_reached:
                return True
        mask = torch.tensor([1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0])
            
        self.out_of_bounds = torch.logical_or(torch.tensor(self.state.robot_state) < torch.tensor(self.robot.state_space.low),
                                              torch.tensor(self.state.robot_state) > torch.tensor(self.robot.state_space.high))
        self.out_of_bounds = torch.any(self.out_of_bounds * mask)
        if self.out_of_bounds.item():
            return True
        return False


  
    
    
 
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