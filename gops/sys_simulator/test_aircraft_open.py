from gops.create_pkg.create_env import create_env
from gops.env.tools.env_dynamic_checker import check_dynamic

                
check_dynamic(env_info={'env_id':'pyth_aircraftconti', 'is_adversary':True,
                        'gamma_atte':5, 'state_threshold':[2.0, 2.0, 2.0], 'max_episode_steps':200}
             )