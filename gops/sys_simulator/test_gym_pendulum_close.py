from gops.create_pkg.create_env import create_env
from gops.env.tools.env_dynamic_checker import check_dynamic


check_dynamic(env_info={'env_id':'gym_pendulum'},
              traj_num=2,
              log_policy_dir='./results/DDPG/gym_pendulum',
              policy_iteration='8000')