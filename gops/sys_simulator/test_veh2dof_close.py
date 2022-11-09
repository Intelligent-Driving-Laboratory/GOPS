from gops.create_pkg.create_env import create_env
from gops.env.tools.env_dynamic_checker import check_dynamic


check_dynamic(env_info={'env_id':'pyth_veh2dofconti', 'pre_horizon':10}, 
              traj_num=1, init_info={"init_state": [[1., 0., 0., 0.]], "ref_time": [0.], "ref_num": [0.]},
              log_policy_dir='./results/INFADP/veh2dofconti_221017-211644',
              policy_iteration='3000')