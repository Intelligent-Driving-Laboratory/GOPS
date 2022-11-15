from gops.create_pkg.create_env import create_env
from gops.env.tools.env_dynamic_checker import check_dynamic, load_args


check_dynamic(env_info=load_args('./results/DDPG/simu_lqs2a1'), 
              traj_num=3,
              init_info={"init_state": [[0.5, -0.3], [1, -1], [-1, 0.7]]},
              log_policy_dir='./results/DDPG/simu_lqs2a1',
              policy_iteration='250000')