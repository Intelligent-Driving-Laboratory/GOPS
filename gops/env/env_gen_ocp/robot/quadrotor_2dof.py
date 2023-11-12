from copy import deepcopy
from gops.env.env_gen_ocp.context.quad_ref_traj import QuadContext
import numpy as np
from gym import spaces
from enum import IntEnum
from enum import Enum
import xml.etree.ElementTree as etxml
import math
import json
from gops.env.env_gen_ocp.env_model.quadrotor_tracking_model import Task,Cost,QuadType
from gops.env.env_gen_ocp.pyth_base import Robot

NAME = 'quadrotor'

INERTIAL_PROP_RAND_INFO = {
    'M': {  # Nominal: 0.027
        'distrib': 'uniform',
        'low': 0.022,
        'high': 0.032
    },
    'Ixx': {  # Nominal: 1.4e-5
        'distrib': 'uniform',
        'low': 1.3e-5,
        'high': 1.5e-5
    },
    'Iyy': {  # Nominal: 1.4e-5
        'distrib': 'uniform',
        'low': 1.3e-5,
        'high': 1.5e-5
    },
    'Izz': {  # Nominal: 2.17e-5
        'distrib': 'uniform',
        'low': 2.07e-5,
        'high': 2.27e-5
    }
}

INIT_STATE_RAND_INFO = {
    'init_x': {
        'distrib': 'uniform',
        'low': -0.5,
        'high': 0.5
    },
    'init_x_dot': {
        'distrib': 'uniform',
        'low': -0.01,
        'high': 0.01
    },
    'init_y': {
        'distrib': 'uniform',
        'low': -0.5,
        'high': 0.5
    },
    'init_y_dot': {
        'distrib': 'uniform',
        'low': -0.01,
        'high': 0.01
    },
    'init_z': {
        'distrib': 'uniform',
        'low': 0.1,
        'high': 1.5
    },
    'init_z_dot': {
        'distrib': 'uniform',
        'low': -0.01,
        'high': 0.01
    },
    'init_phi': {
        'distrib': 'uniform',
        'low': -0.3,
        'high': 0.3
    },
    'init_theta': {
        'distrib': 'uniform',
        'low': -0.3,
        'high': 0.3
    },
    'init_psi': {
        'distrib': 'uniform',
        'low': -0.3,
        'high': 0.3
    },
    'init_p': {
        'distrib': 'uniform',
        'low': -0.01,
        'high': 0.01
    },
    'init_theta_dot': {  # TODO: replace with q.
        'distrib': 'uniform',
        'low': -0.01,
        'high': 0.01
    },
    'init_q': {
        'distrib': 'uniform',
        'low': -0.01,
        'high': 0.01
    },
    'init_r': {
        'distrib': 'uniform',
        'low': -0.01,
        'high': 0.01
    }
}


class Quadrotor():
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
            QuadType.TWO_D: ['init_x', 'init_x_dot', 'init_z', 'init_z_dot', 'init_theta', 'init_theta_dot'],
        }
        self.TASK = Task(task)
        self.COST = Cost(cost)
        if info_mse_metric_state_weight is None:
            self.info_mse_metric_state_weight = np.array([1, 0, 1, 0, 0, 0], ndmin=1, dtype=float)
        else:
            if (self.QUAD_TYPE == QuadType.TWO_D and len(info_mse_metric_state_weight) == 6): 
                self.info_mse_metric_state_weight = np.array(info_mse_metric_state_weight, ndmin=1, dtype=float)
            else:
                raise ValueError('[ERROR] in Quadrotor.__init__(), wrong info_mse_metric_state_weight argument size.')
        
        # Concatenate reference for RL.
        if self.COST == Cost.RL_REWARD and self.TASK == Task.TRAJ_TRACKING and self.obs_goal_horizon > 0:
            # Include future goal state(s).
            # e.g. horizon=1, obs = {state, state_target}
            mul = 1 + self.obs_goal_horizon
            low = np.concatenate([low] * mul)
            high = np.concatenate([high] * mul)
        elif self.COST == Cost.RL_REWARD and self.TASK == Task.STABILIZATION and self.obs_goal_horizon > 0:
            low = np.concatenate([low] * 2)
            high = np.concatenate([high] * 2)
        self.state_dim, self.action_dim = 6, 2
       
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
        self.J = np.array(parameters["J"])
        self.J_INV = np.array(parameters["J_INV"])
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
        self.x_threshold = 2
        self.y_threshold = 2
        self.z_threshold = 2
        self.phi_threshold_radians = 85 * math.pi / 180
        self.theta_threshold_radians = 85 * math.pi / 180
        self.psi_threshold_radians = 180 * math.pi / 180  # Do not bound yaw.

        # Define obs/state bounds, labels and units.
     
       
        # obs/state = {x, x_dot, z, z_dot, theta, theta_dot}.
        low = np.array([
            -self.x_threshold, -np.finfo(np.float32).max,
            self.GROUND_PLANE_Z, -np.finfo(np.float32).max,
            -self.theta_threshold_radians, -np.finfo(np.float32).max
        ])
        high = np.array([
            self.x_threshold, np.finfo(np.float32).max,
            self.z_threshold, np.finfo(np.float32).max,
            self.theta_threshold_radians, np.finfo(np.float32).max
        ])
        self.STATE_LABELS = ['x', 'x_dot', 'z', 'z_dot', 'theta', 'theta_dot']
        self.STATE_UNITS = ['m', 'm/s', 'm', 'm/s', 'rad', 'rad/s']
        # Define the state space for the dynamics.
        self.state_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
    def _set_action_space(self):
        '''Sets the action space of the environment.'''
        # Define action/input dimension, labels, and units.
        # import ipdb ; ipdb.set_trace()
        action_dim = 2
        self.ACTION_LABELS = ['T1', 'T2']
        self.ACTION_UNITS = ['N', 'N'] if not self.NORMALIZED_RL_ACTION_SPACE else ['-', '-']
        n_mot = 4 / action_dim
        a_low = self.KF * n_mot * (self.PWM2RPM_SCALE * self.MIN_PWM + self.PWM2RPM_CONST)**2
        a_high = self.KF * n_mot * (self.PWM2RPM_SCALE * self.MAX_PWM + self.PWM2RPM_CONST)**2
        self.physical_action_bounds = (np.full(action_dim, a_low, np.float32),
                                       np.full(action_dim, a_high, np.float32))

        if self.NORMALIZED_RL_ACTION_SPACE:
            # Normalized thrust (around hover thrust).
            self.hover_thrust = self.GRAVITY_ACC * self.MASS / action_dim
            self.action_space = spaces.Box(low=-np.ones(action_dim),
                                           high=np.ones(action_dim),
                                           dtype=np.float32)
        else:
            # Direct thrust control.
            self.action_space = spaces.Box(low=self.physical_action_bounds[0],
                                           high=self.physical_action_bounds[1],
                                           dtype=np.float32)

    def f_xu(self,X,U):
        m = self.context.MASS
        g= self.GRAVITY_ACC
        u_eq = m * g
   
        #X = np.cat((x, x_dot, z, z_dot, theta, theta_dot), dim=0)
        #U = np.cat((T1, T2))
        self.state_dim, self.action_dim = 6, 2
        X_dot = np.array([X[1],np.sin(X[4]) * (U[0] + U[1]) / m,X[3],np.cos(X[4]) * (U[0] + U[1]) / m - g,
                            X[-1],self.L * (U[1] - U[0]) / self.Iyy / np.sqrt(2.0)])
        return X_dot
       
     
    def reset(self, init_state=None):
        if init_state is None:
            for init_name in INIT_STATE_RAND_INFO:  # Default zero state.
                self.__dict__[init_name.upper()] = 0.
            self.state = np.ones(self.state_dim)
        else:
            if isinstance(init_state, np.ndarray):  # Full state as numpy array .
                for i, init_name in enumerate(self.INIT_STATE_LABELS[self.QUAD_TYPE]):
                    self.__dict__[init_name.upper()] = init_state[i]
            elif isinstance(init_state, dict):  # Partial state as dictionary.
                for init_name in self.INIT_STATE_LABELS[self.QUAD_TYPE]:
                    self.__dict__[init_name.upper()] = init_state.get(init_name, 0.)
            else:
                raise ValueError('[ERROR] in Quadrotor.__init__(), init_state incorrect format.')
        return self._get_obs()
    
    def step(self, thrust):
        X_dot= self.f_xu(X=self.state,U=thrust)
        self.state += self.dt * X_dot
        self.action = thrust
        obs = self._get_obs()
        rew = self._get_reward(thrust)
        done = self._get_done()
        info = self._get_info()
        self.ctrl_step_counter += 1
        return obs, rew, done, info

    def _get_obs(self):
        return self.state
        
    def _get_reward(self,action):
        act_error = action - self.context.U_GOAL
        # Quadratic costs w.r.t state and action
        # TODO: consider using multiple future goal states for cost in tracking
        if self.task == 'STABILIZATION':
            state_error = self.state - self.context.X_GOAL
            dist = np.sum(self.context.rew_state_weight * state_error * state_error)
            dist += np.sum(self.context.rew_act_weight * act_error * act_error)
        if self.task == 'TRAJ_TRACKING':
            wp_idx = min(self.ctrl_step_counter + 1, self.context.X_GOAL.shape[0] - 1)  # +1 because state has already advanced but counter not incremented.
            state_error = self.state - self.context.X_GOAL[wp_idx]
            dist = np.sum(self.context.rew_state_weight * state_error * state_error)
            dist += np.sum(self.context.rew_act_weight * act_error * act_error)
        rew = -dist
        # Convert rew to be positive and bounded [0,1].
        if self.rew_exponential:
            rew = np.exp(rew)
        return rew
   

    def _get_done(self):
        # Done if goal reached for stabilization task with quadratic cost.
        if self.task == 'STABILIZATION' :
            self.goal_reached = bool(np.linalg.norm(self.state - self.context.X_GOAL) < self.context.TASK_INFO['stabilization_goal_tolerance'])
            if self.goal_reached:
                return True 
        # Done if state is out-of-bounds.
        mask = np.array([1, 0, 1, 0, 1, 0])
        # Element-wise or to check out-of-bound conditions.
        # import ipdb; ipdb.set_trace()
        self.out_of_bounds = np.logical_or(self.state < self.state_space.low,
                                        self.state > self.state_space.high)
        # Mask out un-included dimensions (i.e. velocities)
        self.out_of_bounds = np.any(self.out_of_bounds * mask)
        # Early terminate if needed.
        if self.out_of_bounds:
            return True
        return False
      
    def _get_info(self):
        '''Generates the info dictionary returned by every call to .step().

        Returns:
            info (dict): A dictionary with information about the constraints evaluations and violations.
        '''
        info = {}
        if self.task == 'STABILIZATION' :
            info['goal_reached'] = self.goal_reached  # Add boolean flag for the goal being reached.
        info['out_of_bounds'] = self.out_of_bounds
        # Add MSE.
        state = deepcopy(self.state)
        if self.task == 'STABILIZATION':
            state_error = state - self.context.X_GOAL
        elif self.task == 'TRAJ_TRACKING':
            # TODO: should use angle wrapping
            # state[4] = normalize_angle(state[4])
            wp_idx = min(self.ctrl_step_counter + 1, self.context.X_GOAL.shape[0] - 1)  # +1 so that state is being compared with proper reference state.
            state_error = state - self.context.X_GOAL[wp_idx]
        # Filter only relevant dimensions.

        # import ipdb; ipdb.set_trace()
        state_error = state_error * self.info_mse_metric_state_weight
        info['mse'] = np.sum(state_error ** 2)
        # if self.constraints is not None:
        #     info['constraint_values'] = self.constraints.get_values(self)
        #     info['constraint_violations'] = self.constraints.get_violations(self)
        return info
 
