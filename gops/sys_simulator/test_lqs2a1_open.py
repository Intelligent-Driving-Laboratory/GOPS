from gops.create_pkg.create_env import create_env
from gops.env.tools.env_dynamic_checker import check_dynamic


check_dynamic(env_info={'env_id':'pyth_lq', 'lq_config':'s2a1'}, 
              traj_num=2,
              init_info={"init_state": [[0.5, -0.5], [1, -1]]})