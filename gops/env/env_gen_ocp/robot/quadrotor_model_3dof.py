from typing import Optional, Sequence

import torch
from gops.env.env_gen_ocp.env_model.pyth_base_model import RobotModel
from gops.env.env_gen_ocp.context.quad_ref_traj import QuadContext
from gops.env.env_gen_ocp.env_model.quadrotor_1dof_tracking_stablization_model import Task,Cost,QuadType



class QuadrotorDynMd3(RobotModel):
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
        self.QUAD_TYPE = QuadType(quad_type)
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
            self.info_mse_metric_state_weight = torch.tensor([1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        else:
            if (len(info_mse_metric_state_weight) == 12):
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
        
        
    def get_next_state(self,X,U):
        m = self.context.MASS
        g= self.GRAVITY_ACC
        u_eq = m * g
    
        self.state_dim, self.action_dim = 12, 4
        J = torch.tensor([[self.Ixx, 0.0, 0.0],
                        [0.0, self.Iyy, 0.0],
                        [0.0, 0.0, self.Izz]])
        Jinv = torch.tensor([[1.0 / self.Ixx, 0.0, 0.0],
                            [0.0, 1.0 / self.Iyy, 0.0],
                            [0.0, 0.0, 1.0 / self.Izz]])
        # gamma = self.KM / self.KF    ## gamma 是电机的转矩常数 KM 和推力常数 KF 的比值。
        gamma = 0.1
        # X = torch.cat((x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body))
        # U = torch.cat((f1, f2, f3, f4))
        def RotZ(psi):
            '''Rotation matrix about Z axis following SDFormat http://sdformat.org/tutorials?tut=specify_pose&cat=specification&.

            Args:
            psi: Scalar rotation

            Returns:
            R: torch Rotation matrix
            '''
            R = torch.tensor([[torch.cos(psi), -torch.sin(psi), 0],
                            [torch.sin(psi), torch.cos(psi), 0],
                            [0, 0, 1]])
            return R

        def RotY(theta):
            '''Rotation matrix about Y axis following SDFormat http://sdformat.org/tutorials?tut=specify_pose&cat=specification&.

            Args:
            theta: Scalar rotation

            Returns:
            R: torch Rotation matrix
            '''
            R = torch.tensor([[torch.cos(theta), 0, torch.sin(theta)],
                            [0, 1, 0],
                            [-torch.sin(theta), 0, torch.cos(theta)]])
            return R

        def RotX(phi):
            '''Rotation matrix about X axis following SDFormat http://sdformat.org/tutorials?tut=specify_pose&cat=specification&.

            Args:
            phi: Scalar rotation

            Returns:
            R: torch Rotation matrix
            '''
            R = torch.tensor([[1, 0, 0],
                            [0, torch.cos(phi), -torch.sin(phi)],
                            [0, torch.sin(phi), torch.cos(phi)]])
            return R

        Rob = RotZ(X[6]) @ RotY(X[7]) @ RotX(X[8])
        # import ipdb;ipdb.set_trace()
        pos_ddot = Rob @ torch.concatenate((torch.tensor([0.0]), torch.tensor([0.0]), (U[0] + U[1] + U[2] + U[3]).reshape(1,))) / m - torch.concatenate((torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([g])))
        pos_dot = torch.tensor([X[1],X[3],X[5]])
        Mb = torch.tensor((self.L / torch.sqrt(2.0) * (U[0] + U[1] - U[2] - U[3]),
                        self.L / torch.sqrt(torch.tensor(2.0)) * (-U[0] + U[1] + U[2] - U[3]),
                        gamma * (-U[0] + U[1] - U[2] + U[3])))
        def skew_matrix(angular_velocity):
            # Create a 3x3 skew-symmetric matrix from a 3D angular velocity vector
            return torch.tensor([[0, -angular_velocity[2], angular_velocity[1]],
                                [angular_velocity[2], 0, -angular_velocity[0]],
                                [-angular_velocity[1], angular_velocity[0], 0]])
        rate_dot = Jinv @ (Mb - skew_matrix(torch.tensor([X[9], X[10], X[11]])) @ J @ torch.tensor([X[9], X[10], X[11]]))
        # Define the components of the rotation matrix
        R1 = torch.tensor([[1.0, 0.0, 0.0],
                        [0.0, torch.cos(X[6]), -torch.sin(X[6])],
                        [0.0, torch.sin(X[6]), torch.cos(X[6])]])
        R2 = torch.tensor([[torch.cos(X[7]), 0.0, torch.sin(X[7])],
                        [0.0, 1.0, 0.0],
                        [-torch.sin(X[7]), 0.0, torch.cos(X[7])]])
        R3 = torch.tensor([[torch.cos(X[8]), -torch.sin(X[8]), 0.0],
                        [torch.sin(X[8]), torch.cos(X[8]), 0.0],
                        [0.0, 0.0, 1.0]])
        # Compute the angular velocity vector
        ang_dot = R1 @ R2 @ R3 @ torch.tensor((X[9], X[10], X[11]))
        # Flatten ang_dot and rate_dot into one-dimensional tensors
        ang_dot_flat = ang_dot.reshape(-1)
        rate_dot_flat = rate_dot.reshape(-1)
        # Concatenate all tensors
        X_dot = torch.concatenate((pos_dot, pos_ddot, ang_dot_flat, rate_dot_flat))
        return X_dot
    