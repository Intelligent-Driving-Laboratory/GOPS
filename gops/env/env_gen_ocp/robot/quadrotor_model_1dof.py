from typing import Optional, Sequence

import torch
from gops.env.env_gen_ocp.env_model.pyth_base_model import RobotModel
from gops.env.env_gen_ocp.context.quad_ref_traj import QuadContext
from gops.env.env_gen_ocp.env_model.quadrotor_tracking_model import Task,Cost,QuadType
from gops.env.env_gen_ocp.env_model.quadrotor_tracking_model import Task,Cost,QuadType


class QuadrotorDynMdl(RobotModel):
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
        
        
    def get_next_state(self,X,U):
        m = self.context.MASS
        g= self.GRAVITY_ACC
        u_eq = m * g
        self.state_dim, self.action_dim = 2, 1
        X_dot = torch.tensor([X[1], U[0] / m - g])  
        return X_dot
       