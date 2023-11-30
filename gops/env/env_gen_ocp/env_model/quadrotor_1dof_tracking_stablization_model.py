from typing import Dict, Optional, Tuple, Union
from gym import spaces

import torch
from enum import IntEnum,Enum
from gops.env.env_gen_ocp.env_model.pyth_base_model import EnvModel
from gops.env.env_gen_ocp.pyth_base import State
from gops.env.env_gen_ocp.context.quad_ref_traj import QuadContext
from gops.env.env_gen_ocp.robot.quadrotor_model_1dof import QuadrotorDynMdl
import numpy as np
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
    
class Quadrotor1dofTrackingStablizationModel(EnvModel):

    def __init__(
        self,     
        task = 'TRAJ_TRACKING',
        device: Union[torch.device, str, None] = None,
        **kwargs,
    ):
        dt = 0.01
        self.obs_dim = 4
        action_dim = 1
        dt=dt
        self.robot_model = QuadrotorDynMdl()
        super().__init__(
            obs_lower_bound=None,
            obs_upper_bound=None,
            action_lower_bound=-torch.ones(action_dim),
            action_upper_bound= torch.ones(action_dim),
            device = device,
        )
        self.robot: QuadrotorDynMdl = QuadrotorDynMdl(
            quad_type = QuadType.ONE_D,
        )
        self.context: QuadContext = QuadContext(
            quad_type = QuadType.ONE_D
        )
        self.state_space = self.robot.state_space
        self.action_space = self.robot.action_space
        self.max_episode_steps = 200
        self.task = task
        
     
    # def get_next_state(self, state: State, action: np.ndarray) -> State[np.ndarray]:
    #     return State(
    #         robot_state=self.robot.get_next_state(state = state.robot_state, action = action),
    #         context_state = self.context.step()
    #     )
        
    def reset(self) -> Tuple[torch.Tensor, dict]:
        numpy_state = self.robot.reset()
        tensor_state = torch.from_numpy(numpy_state).float()
        self._state = {
            'robot_state': tensor_state,
            'context_state': self.context.state  
        }
        return tensor_state, self._state
 

    def get_obs(self,state) -> torch.Tensor:
        t = state.context_state.t
        # return torch.from_numpy(self._state.robot_state).float() if isinstance(self._state.robot_state, np.ndarray) else self._state.robot_state
        # return torch.squeeze(torch.concat(( torch.tensor(state.context_state.reference[:,t]), state.robot_state),1))
        return state.robot_state
    def get_reward(self, state:torch.Tensor , action: torch.Tensor) -> float:
        act_error = action - torch.tensor(self.context.U_GOAL)
        # state = deepcopy(self.s)
        if self.task == 'STABILIZATION':
            state_error = torch.tensor(state.robot_state) - torch.tensor(state.context_state.reference[state.context_state.t])
            dist = torch.sum(torch.tensor(self.context.rew_state_weight) * state_error ** 2)
            dist += torch.sum(torch.tensor(self.context.rew_act_weight) * act_error ** 2)
        elif self.task == 'TRAJ_TRACKING':
            wp_idx = min(self.robot.ctrl_step_counter , self.context.X_GOAL.shape[0] - 1)  # +1 because state has already advanced but counter not incremented.
            state_error = state.robot_state - torch.tensor(self.context.X_GOAL[wp_idx]).unsqueeze(0)
            dist = torch.sum(torch.tensor(self.context.rew_state_weight) * state_error ** 2)
            dist += torch.sum(torch.tensor(self.context.rew_act_weight) * act_error ** 2)
            # state_error = torch.tensor(state.robot_state) - torch.tensor(state.context_state.reference)
            # dist = torch.sum(torch.tensor(self.context.rew_state_weight) * state_error ** 2)
            # dist += torch.sum(torch.tensor(self.context.rew_act_weight) * act_error ** 2)
        rew = -dist
        # rew = torch.tensor(-dist.item())
        # if self.robot.rew_exponential:
        #     rew = torch.exp(-dist)
        return rew

    def get_terminated(self,state) -> bool:
        if self.task == 'STABILIZATION':
            self.goal_reached = (torch.linalg.norm(torch.tensor(state.robot_state) - torch.tensor(state.context_state.reference)) < self.context.TASK_INFO['stabilization_goal_tolerance']).item()
            if self.goal_reached:
                return True
        
        # mask = torch.tensor([1, 0])
        out_of_bounds = torch.logical_or(torch.tensor(state.robot_state) < torch.tensor(self.robot.state_space.low),
                                              torch.tensor(state.robot_state) > torch.tensor(self.robot.state_space.high))
        out_of_bounds = torch.any(out_of_bounds , dim=1)
        # if self.out_of_bounds.item():
        #     return True
        return out_of_bounds
    
 
def env_creator(**kwargs):
    return Quadrotor1dofTrackingStablizationModel(**kwargs)

if __name__ == "__main__":
    from gops.env.env_gen_ocp.veh3dof_tracking import Veh3DoFTracking
    # from gops.env.inspector.consistency_checker import check_env_model_consistency
    env = Quadrotor1dofTrackingStablizationModel(2002)
    model = QuadrotorDynMdl()
    # check_env_model_consistency(env, model)
# if __name__ == "__main__":
#     # test consistency with old environment
#     # for quad_type in QuadType:  
#     #     import configparser
#     #     import os
#     #     config = configparser.ConfigParser()
#     #     config['quad_type'] = {'quad_type': str(QuadType.ONE_D)}
#     #     with open('./config.ini', 'w') as configfile:
#     #         config.write(configfile)
#     #     print('\n----------quad_type:',quad_type,'----------')
#     env_new = QuadTracking(quad_type = QuadType.THREE_D)
#     seed = 1
#     env_new.seed(seed)
#     torch.random.seed(seed)
#     obs_new = env_new.reset()
#     print("reset obs close:", obs_new)
#     action = torch.random.random(4)
#     state = env_new.get_next_state(obs_new,action)
#     print("step reward close:",  state)