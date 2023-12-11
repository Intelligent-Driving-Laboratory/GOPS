from typing import Dict, Optional, Sequence, Tuple
import numpy as np
from gym import spaces
from enum import IntEnum
from copy import deepcopy
from gops.env.env_gen_ocp.pyth_base import Env, State
from gops.env.env_gen_ocp.robot.quadrotor_1dof import Quadrotor
from gops.env.env_gen_ocp.context.quad_ref_traj import QuadContext


class QuadType(IntEnum):
    '''Quadrotor types numeration class.'''
    ONE_D = 1  # One-dimensional (along z) movement.
    TWO_D = 2  # Two-dimensional (in the x-z plane) movement.
    THREE_D = 3  # Three-dimensional movement.
    
class Quadrotor1dofTrackingStablization(Env):
    def __init__(
        self,       
        task = "TRAJ_TRACKING",
        *,
        pre_horizon: int = 10,
        **kwargs,
    ):
        self.robot: Quadrotor = Quadrotor(
        )
        self.context: QuadContext = QuadContext(
            pre_horizon=pre_horizon,
            quad_type = QuadType.ONE_D
        )
        self.observation_space =  self.robot.state_space
        
        # self.observation_space = spaces.Box(
        #     low=np.tile(self.robot.state_space.low, 2),
        #     high=np.tile(self.robot.state_space.high, 2),
        #     dtype=np.float32,
        # )
        self.action_space = self.robot.action_space
        self.max_episode_steps = 200
        self.seed()
        self.task = task

    def reset(
        self,
        init_state = None
    ) -> Tuple[np.ndarray, dict]:
        self._state = State(
            robot_state=self.robot.reset(init_state),
            context_state= self.context.reset()
        )
        return self._get_obs(),self._get_info()

    def _get_obs(self) -> np.ndarray:
        # return np.concatenate((self._state.context_state.reference[0].reshape(1, -1),self._state.robot_state.reshape(1, -1)),axis=0).flatten()
        return self._state.robot_state
    def _get_reward(self, action: np.ndarray) -> float:
        act_error = action - self.context.U_GOAL
        # Quadratic costs w.r.t state and action
        # TODO: consider using multiple future goal states for cost in tracking
        if self.task == 'STABILIZATION':
            wp_idx = min(self.robot.ctrl_step_counter + 1, self.context.X_GOAL.shape[0] - 1)  # +1 because state has already advanced but counter not incremented.
            state_error = self._state.robot_state - self.context.X_GOAL[wp_idx]
            dist = np.sum(self.context.rew_state_weight * state_error * state_error)
            # dist += np.sum(self.context.rew_act_weight * act_error * act_error)
            # dist = np.sum(act_error * act_error)
        if self.task == 'TRAJ_TRACKING':
            wp_idx = min(self.robot.ctrl_step_counter, self.context.X_GOAL.shape[0] - 1)  # +1 because state has already advanced but counter not incremented.
            state_error = self._state.robot_state - self.context.X_GOAL[wp_idx]
            dist = np.sum(self.context.rew_state_weight * state_error * state_error)
            dist += np.sum(self.context.rew_act_weight * act_error * act_error)
        rew = -dist
        # Convert rew to be positive and bounded [0,1].
        # if self.robot.rew_exponential:
        #     rew = np.exp(rew)
        return rew

    def _get_terminated(self) -> bool:
           
        # Done if goal reached for stabilization task with quadratic cost.
        if self.task == 'STABILIZATION' :
            wp_idx = min(self.robot.ctrl_step_counter, self.context.X_GOAL.shape[0] - 1)  # +1 because state has already advanced but counter not incremented.
            state_error = self._state.robot_state - self.context.X_GOAL[wp_idx]
            self.goal_reached = bool(np.linalg.norm(state_error) < self.context.TASK_INFO['stabilization_goal_tolerance'])
            # if self.goal_reached:
            #     return True 
        # Done if state is out-of-bounds.
        mask = np.array([1, 1])
      
        # Element-wise or to check out-of-bound conditions.
        # import ipdb; ipdb.set_trace()
        self.out_of_bounds = np.logical_or(self._state.robot_state < self.robot.state_space.low,
                                        self._state.robot_state > self.robot.state_space.high)
        # Mask out un-included dimensions (i.e. velocities)
        self.out_of_bounds = np.any(self.out_of_bounds * mask)
        # Early terminate if needed.
        if self.out_of_bounds:
            return True
        return False
    
    def _get_info(self) -> dict:
       
        '''Generates the info dictionary returned by every call to .step().

        Returns:
            info (dict): A dictionary with information about the constraints evaluations and violations.
        '''
        info = {}
        # if self.task == 'STABILIZATION' :
        #     info['goal_reached'] = self.goal_reached  # Add boolean flag for the goal being reached.
        # info['out_of_bounds'] = self.out_of_bounds
        # # Add MSE.
        # state = deepcopy(self._state)
        # if self.task == 'STABILIZATION':
        #     state_error = state.robot_state - state.context_state.reference
        # elif self.task == 'TRAJ_TRACKING':
        #     # TODO: should use angle wrapping
        #     # state[4] = normalize_angle(state[4])
        #     # wp_idx = min(self.robot.ctrl_step_counter + 1, self._state.context_state.shape[0] - 1)  # +1 so that state is being compared with proper reference state.
        #     state_error = state.robot_state - state.context_state.reference
        # # Filter only relevant dimensions.
        # # import ipdb; ipdb.set_trace()
        # state_error = state_error * self.robot.info_mse_metric_state_weight
        # info['mse'] = np.sum(state_error ** 2)
        # # if self.constraints is not None:
        # #     info['constraint_values'] = self.constraints.get_values(self)
        # #     info['constraint_violations'] = self.constraints.get_violations(self)
        info.update(super()._get_info())
        return info

    def render(self, mode="human"):
        pass
       
        
def env_creator(**kwargs):
    return Quadrotor1dofTrackingStablization(task = 'TRAJ_TRACKING', **kwargs)


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
    env = Quadrotor1dofTrackingStablization()
    # seed = 1
    # env_new.seed(seed)
    # np.random.seed(seed)
    
    obs, info = env.reset()
    actions = [0.116961166, 0.35555518, 0.23784652, 0.3495553, 0.18962601, 0.33521265, 0.22090153, 0.44335434, 0.19350258, 0.15874778]

    for i in range(10):
        action = np.array([actions[i]])
        print(f'-----------{i}------------')
        print('obs',obs)
        print('act',action)
        
        obs_, rew, done, info = env.step(action=action)
        print('context',info['state'].context_state.reference[0])
        print('rew',rew)
        print('obs_',obs_)
        obs = obs_

    


