from typing import Optional, Sequence
import json
import torch
from gops.env.env_gen_ocp.env_model.pyth_base_model import RobotModel
from gops.env.env_gen_ocp.context.quad_ref_traj import QuadContext
from gym import spaces
from enum import IntEnum,Enum

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
    
class QuadrotorDynMdl(RobotModel):
    def __init__(self, 
                 prior_prop={},
                 obs_goal_horizon = 0,
                 rew_exponential=True,  
                 init_state=None,
                 task: Task = Task.STABILIZATION,
                 cost: Cost = Cost.RL_REWARD,
                 info_mse_metric_state_weight=None,
                 **kwargs ):
      
        self.obs_goal_horizon = obs_goal_horizon
        self.init_state = init_state
        # self.QUAD_TYPE = QuadType(quad_type)
        self.L = 1.
        self.rew_exponential = rew_exponential
        self.GRAVITY_ACC = 9.81
        self.CTRL_TIMESTEP = 0.01 
        self.TIMESTEP = 0.001  
        self.state = None
        self.dt = self.TIMESTEP
        self.x_threshold = 2
        self.context = QuadContext()
        self.ctrl_step_counter = 0 
        self.task = self.context.task
        self.GROUND_PLANE_Z = -0.05
       
        self.STATE_LABELS = ['x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot',
                                'phi', 'theta', 'psi', 'p', 'q', 'r']
        self.STATE_UNITS = ['m', 'm/s', 'm', 'm/s', 'm', 'm/s',
                            'rad', 'rad', 'rad', 'rad/s', 'rad/s', 'rad/s']
        self.INIT_STATE_LABELS = {
            QuadType.ONE_D: ['init_x', 'init_x_dot']
        }
        self.TASK = Task(task)
        self.COST = Cost(cost)
        if info_mse_metric_state_weight is None:
            self.info_mse_metric_state_weight = torch.tensor([1, 0],  dtype=float)
        else:
            if (len(info_mse_metric_state_weight) == 2) :
                self.info_mse_metric_state_weight = torch.tensor(info_mse_metric_state_weight, ndmin=1, dtype=float)
            else:
                raise ValueError('[ERROR] in Quadrotor.__init__(), wrong info_mse_metric_state_weight argument size.')
        
        # Concatenate reference for RL.
        if self.COST == Cost.RL_REWARD and self.TASK == Task.TRAJ_TRACKING and self.obs_goal_horizon > 0:
            # Include future goal state(s).
            # e.g. horizon=1, obs = {state, state_target}
            mul = 1 + self.obs_goal_horizon
            low = torch.concatenate([low] * mul)
            high = torch.concatenate([high] * mul)
        elif self.COST == Cost.RL_REWARD and self.TASK == Task.STABILIZATION and self.obs_goal_horizon > 0:
            low = torch.concatenate([low] * 2)
            high = torch.concatenate([high] * 2)

        self.state_dim, self.action_dim = 2, 1
        self.NORMALIZED_RL_ACTION_SPACE = True
      
        
        # Define obs space exposed to the controller.
        # Note how the obs space can differ from state space (i.e. augmented with the next reference states for RL)
        self.URDF_PATH = '/home/qinshentao/code/gops/gops/env/env_gen_ocp/robot/quadrotor_parm.json'
        self.load_parameters()
        self.Iyy = prior_prop.get('Iyy', self.J[1, 1])
        self.Ixx = prior_prop.get('Ixx', self.J[0, 0])
        self.Izz = prior_prop.get('Izz', self.J[2, 2])
        self._set_action_space()
        self._set_observation_space()
        
    def load_parameters(self):
        with open(self.URDF_PATH, 'r') as f:
            parameters = json.load(f)
        self.MASS = parameters["MASS"]
        self.L = parameters["L"]
        self.THRUST2WEIGHT_RATIO = parameters["THRUST2WEIGHT_RATIO"]
        self.J = torch.tensor(parameters["J"])
        self.J_INV = torch.tensor(parameters["J_INV"])
        self.KF = parameters["KF"]
        self.KM = parameters["KM"]
        self.COLLISION_H = parameters["COLLISION_H"]
        self.COLLISION_R = parameters["COLLISION_R"]
        self.COLLISION_Z_OFFSET = parameters["COLLISION_Z_OFFSET"]
        self.MAX_SPEED_KMH = parameters["MAX_SPEED_KMH"]
        self.GND_EFF_COEFF = parameters["GND_EFF_COEFF"]
        self.PROP_RADIUS = parameters["PROP_RADIUS"]
        self.DRAG_COEFF = parameters["DRAG_COEFF"]
        self.DW_COEFF_1 = parameters["DW_COEFF_1"]
        self.DW_COEFF_2 = parameters["DW_COEFF_2"]
        self.DW_COEFF_3 = parameters["DW_COEFF_3"]
        self.PWM2RPM_SCALE = parameters["PWM2RPM_SCALE"]
        self.PWM2RPM_CONST = parameters["PWM2RPM_CONST"]
        self.MIN_PWM = parameters["MIN_PWM"]
        self.MAX_PWM = parameters["MAX_PWM"]
     
    def _set_observation_space(self):
        '''Sets the observation space of the environment.'''
        self.z_threshold = 2
      

        # Define obs/state bounds, labels and units.
        # obs/state = {z, z_dot}.
        import numpy as np
        low = np.array([self.GROUND_PLANE_Z, -np.finfo(np.float32).max])
        high = np.array([self.z_threshold, np.finfo(np.float32).max])
        self.STATE_LABELS = ['z', 'z_dot']
        self.STATE_UNITS = ['m', 'm/s']
      
        self.state_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
    def _set_action_space(self):
        '''Sets the action space of the environment.'''
        # Define action/input dimension, labels, and units.
        # import ipdb ; ipdb.set_trace()
        action_dim = 1
        n_mot = 4 / action_dim
        a_low = self.KF * n_mot * (self.PWM2RPM_SCALE * self.MIN_PWM + self.PWM2RPM_CONST)**2
        a_high = self.KF * n_mot * (self.PWM2RPM_SCALE * self.MAX_PWM + self.PWM2RPM_CONST)**2
        self.physical_action_bounds = (torch.full((action_dim,), a_low, dtype=torch.float32),
                                    torch.full((action_dim,), a_high, dtype=torch.float32))

        # Normalized thrust action space (around hover thrust).
        import numpy as np
        self.hover_thrust = self.GRAVITY_ACC * self.MASS / action_dim
        self.action_space = spaces.Box(low=-np.ones(action_dim),
                                        high=np.ones(action_dim),
                                        dtype=np.float32)
        # # else, Direct thrust control.
        # self.action_space = spaces.Box(low=self.physical_action_bounds[0],
        #                                 high=self.physical_action_bounds[1],
        #                                 dtype=torch.float32)
    def get_next_state(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        m = self.context.MASS
        g= self.GRAVITY_ACC
        # u_eq = m * g
        self.state_dim, self.action_dim = 2, 1
        X_dot = torch.stack([state[:, 1], action[:, 0] / m - g], dim=1) 
        next_state = state + self.dt * X_dot
        self.ctrl_step_counter += 1
        return next_state
       