from copy import deepcopy
from gops.env.env_gen_ocp.context.quad_ref_traj import QuadContext
import numpy as np
from gym import spaces
import xml.etree.ElementTree as etxml
import math
import json
from gops.env.env_gen_ocp.env_model.quadrotor_1dof_tracking_stablization_model import Task,Cost,QuadType
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
                 quad_type = None,
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
            QuadType.THREE_D: ['init_x', 'init_x_dot', 'init_y', 'init_y_dot', 'init_z', 'init_z_dot',
                               'init_phi', 'init_theta', 'init_psi', 'init_p', 'init_q', 'init_r']
        }
        self.TASK = Task(task)
        self.COST = Cost(cost)
        if info_mse_metric_state_weight is None:
            self.info_mse_metric_state_weight = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], ndmin=1, dtype=float)
        else:
            if  len(info_mse_metric_state_weight) == 12:
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
        self.state_dim, self.action_dim = 12, 4
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
       
       
            # obs/state = {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body}.
        low = np.array([
            -self.x_threshold, -np.finfo(np.float32).max,
            -self.y_threshold, -np.finfo(np.float32).max,
            self.GROUND_PLANE_Z, -np.finfo(np.float32).max,
            -self.phi_threshold_radians, -self.theta_threshold_radians, -self.psi_threshold_radians,
            -np.finfo(np.float32).max, -np.finfo(np.float32).max, -np.finfo(np.float32).max
        ])
        high = np.array([
            self.x_threshold, np.finfo(np.float32).max,
            self.y_threshold, np.finfo(np.float32).max,
            self.z_threshold, np.finfo(np.float32).max,
            self.phi_threshold_radians, self.theta_threshold_radians, self.psi_threshold_radians,
            np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max
        ])
        self.STATE_LABELS = ['x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot',
                            'phi', 'theta', 'psi', 'p', 'q', 'r']
        self.STATE_UNITS = ['m', 'm/s', 'm', 'm/s', 'm', 'm/s',
                            'rad', 'rad', 'rad', 'rad/s', 'rad/s', 'rad/s']
        # Define the state space for the dynamics.
        self.state_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
    def _set_action_space(self):
        '''Sets the action space of the environment.'''
        # Define action/input dimension, labels, and units.
        # import ipdb ; ipdb.set_trace()
     
       
        action_dim = 4
        self.ACTION_LABELS = ['T1', 'T2', 'T3', 'T4']
        self.ACTION_UNITS = ['N', 'N', 'N', 'N'] if not self.NORMALIZED_RL_ACTION_SPACE else ['-', '-', '-', '-']

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
       
        self.state_dim, self.action_dim = 12, 4
        J = np.array([[self.Ixx, 0.0, 0.0],
                        [0.0, self.Iyy, 0.0],
                        [0.0, 0.0, self.Izz]])
        Jinv = np.array([[1.0 / self.Ixx, 0.0, 0.0],
                            [0.0, 1.0 / self.Iyy, 0.0],
                            [0.0, 0.0, 1.0 / self.Izz]])
        # gamma = self.KM / self.KF    ## gamma 是电机的转矩常数 KM 和推力常数 KF 的比值。
        gamma = 0.1
        # X = np.cat((x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body))
        # U = np.cat((f1, f2, f3, f4))
        def RotZ(psi):
            '''Rotation matrix about Z axis following SDFormat http://sdformat.org/tutorials?tut=specify_pose&cat=specification&.

            Args:
            psi: Scalar rotation

            Returns:
            R: torch Rotation matrix
            '''
            R = np.array([[np.cos(psi), -np.sin(psi), 0],
                            [np.sin(psi), np.cos(psi), 0],
                            [0, 0, 1]])
            return R

        def RotY(theta):
            '''Rotation matrix about Y axis following SDFormat http://sdformat.org/tutorials?tut=specify_pose&cat=specification&.

            Args:
            theta: Scalar rotation

            Returns:
            R: torch Rotation matrix
            '''
            R = np.array([[np.cos(theta), 0, np.sin(theta)],
                            [0, 1, 0],
                            [-np.sin(theta), 0, np.cos(theta)]])
            return R

        def RotX(phi):
            '''Rotation matrix about X axis following SDFormat http://sdformat.org/tutorials?tut=specify_pose&cat=specification&.

            Args:
            phi: Scalar rotation

            Returns:
            R: torch Rotation matrix
            '''
            R = np.array([[1, 0, 0],
                            [0, np.cos(phi), -np.sin(phi)],
                            [0, np.sin(phi), np.cos(phi)]])
            return R

        Rob = RotZ(X[6]) @ RotY(X[7]) @ RotX(X[8])
        # import ipdb;ipdb.set_trace()
        pos_ddot = Rob @ np.concatenate((np.array([0.0]), np.array([0.0]), (U[0] + U[1] + U[2] + U[3]).reshape(1,))) / m - np.concatenate((np.array([0.0]), np.array([0.0]), np.array([g])))
        pos_dot = np.array([X[1],X[3],X[5]])
        Mb = np.array((self.L / np.sqrt(2.0) * (U[0] + U[1] - U[2] - U[3]),
                        self.L / np.sqrt(np.array(2.0)) * (-U[0] + U[1] + U[2] - U[3]),
                        gamma * (-U[0] + U[1] - U[2] + U[3])))
        def skew_matrix(angular_velocity):
            # Create a 3x3 skew-symmetric matrix from a 3D angular velocity vector
            return np.array([[0, -angular_velocity[2], angular_velocity[1]],
                                [angular_velocity[2], 0, -angular_velocity[0]],
                                [-angular_velocity[1], angular_velocity[0], 0]])
        rate_dot = Jinv @ (Mb - skew_matrix(np.array([X[9], X[10], X[11]])) @ J @ np.array([X[9], X[10], X[11]]))
        # Define the components of the rotation matrix
        R1 = np.array([[1.0, 0.0, 0.0],
                        [0.0, np.cos(X[6]), -np.sin(X[6])],
                        [0.0, np.sin(X[6]), np.cos(X[6])]])
        R2 = np.array([[np.cos(X[7]), 0.0, np.sin(X[7])],
                        [0.0, 1.0, 0.0],
                        [-np.sin(X[7]), 0.0, np.cos(X[7])]])
        R3 = np.array([[np.cos(X[8]), -np.sin(X[8]), 0.0],
                        [np.sin(X[8]), np.cos(X[8]), 0.0],
                        [0.0, 0.0, 1.0]])
        # Compute the angular velocity vector
        ang_dot = R1 @ R2 @ R3 @ np.array((X[9], X[10], X[11]))
        # Flatten ang_dot and rate_dot into one-dimensional tensors
        ang_dot_flat = ang_dot.reshape(-1)
        rate_dot_flat = rate_dot.reshape(-1)
        # Concatenate all tensors
        X_dot = np.concatenate((pos_dot, pos_ddot, ang_dot_flat, rate_dot_flat))
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
        return self.state
    
    def step(self, thrust):
        X_dot= self.f_xu(X=self.state,U=thrust)
        self.state += self.dt * X_dot
        self.action = thrust

        self.ctrl_step_counter += 1
        return X_dot

   