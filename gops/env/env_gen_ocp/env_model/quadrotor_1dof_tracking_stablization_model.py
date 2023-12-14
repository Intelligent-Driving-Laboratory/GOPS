from typing import Union

import torch
from enum import IntEnum,Enum
from gops.env.env_gen_ocp.env_model.pyth_base_model import EnvModel
from gops.env.env_gen_ocp.context.quad_ref_traj import QuadContext
from gops.env.env_gen_ocp.robot.quadrotor_model_1dof import QuadrotorDynMdl

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
        # self.obs_dim = 4
        self.obs_dim = 2
        action_dim = 1
        self.robot_model = QuadrotorDynMdl()
        super().__init__(
            obs_lower_bound=[0, -2],
            obs_upper_bound=[2, 2],
            action_lower_bound=[0] * action_dim,
            action_upper_bound=[20] * action_dim,
            device = device,
        )
        self.robot: QuadrotorDynMdl = QuadrotorDynMdl(
            quad_type = QuadType.ONE_D,
        )
        self.context: QuadContext = QuadContext(
            quad_type = QuadType.ONE_D
        )
        self.max_episode_steps = 200
        self.task = task

    def get_obs(self,state) -> torch.Tensor:
        t = state.context_state.t
        # return torch.from_numpy(self._state.robot_state).float() if isinstance(self._state.robot_state, np.ndarray) else self._state.robot_state
        # return torch.squeeze(torch.concat(( torch.tensor(state.context_state.reference[:,t]), state.robot_state),1))
        return state.robot_state
    def get_reward(self, state:torch.Tensor , action: torch.Tensor) -> float:
        act_error = action - torch.tensor(self.context.U_GOAL)
        # state = deepcopy(self.s)
    
        if self.task == 'STABILIZATION':
            state_error = state.robot_state - state.context_state.reference[:,0,:]
            dist = torch.sum(torch.tensor(self.context.rew_state_weight) * state_error ** 2,dim=1)
            # dist += torch.sum(torch.tensor(self.context.rew_act_weight) * act_error ** 2,dim=1)
            # dist = torch.sum(act_error ** 2,dim=1)
            
        elif self.task == 'TRAJ_TRACKING':
            t = state.context_state.t
            state_error = state.robot_state - state.context_state.reference[:,t,:]
            dist = torch.sum(torch.tensor(self.context.rew_state_weight) * state_error ** 2,dim=1)
            dist += torch.sum(torch.tensor(self.context.rew_act_weight) * act_error ** 2,dim=1)
            # state_error = torch.tensor(state.robot_state) - torch.tensor(state.context_state.reference)
            # dist = torch.sum(torch.tensor(self.context.rew_state_weight) * state_error ** 2)
            # dist += torch.sum(torch.tensor(self.context.rew_act_weight) * act_error ** 2)
        rew = -dist
        # rew = torch.tensor(-dist.item())
        # if self.robot.rew_exponential:
        #     rew = torch.exp(-dist)
        return rew

    def get_terminated(self,state) -> bool:
        wp_idx = min(self.robot.ctrl_step_counter + 1, self.context.X_GOAL.shape[0] - 1)
        if self.task == 'STABILIZATION':
            self.goal_reached = (torch.linalg.norm(
                state.robot_state - torch.tensor(state.context_state.reference[:,0,:]),dim=1
                ) < self.context.TASK_INFO['stabilization_goal_tolerance'])
        
        # mask = torch.tensor([1, 0])
        out_of_bounds = torch.logical_or(state.robot_state < self.obs_lower_bound,
                                              state.robot_state > self.obs_upper_bound)
        out_of_bounds = torch.any(out_of_bounds , dim=1)
        # if self.out_of_bounds.item():
        #     return True
        return out_of_bounds
    
 
def env_creator(**kwargs):
    return Quadrotor1dofTrackingStablizationModel(**kwargs)
