from gops.utils.common_utils import get_args_from_json
from sys_run import PolicyRunner
import torch
from gops.algorithm.infadp import ApproxContainer
import os
import argparse


def load_args(log_policy_dir):
    json_path = os.path.join(log_policy_dir, "config.json")
    parser = argparse.ArgumentParser()
    args_dict = vars(parser.parse_args())
    args = get_args_from_json(json_path, args_dict)
    return args

def load_policy(log_policy_dir, trained_policy_iteration):
    # Create policy
    args = load_args(log_policy_dir)
    networks = ApproxContainer(**args)

    # Load trained policy
    log_path = log_policy_dir + "/apprfunc/apprfunc_{}.pkl".format(trained_policy_iteration)
    networks.load_state_dict(torch.load(log_path))
    return networks

def terminal_cost(obs):
    value_net = load_policy("code/gops/results/INFADP/s3a1", '20000').v
    return -value_net(obs)

runner = PolicyRunner(
    log_policy_dir_list=["code/gops/results/INFADP/s5a1"]*2,
    trained_policy_iteration_list=['3000', '190000'],
    is_init_info=True,
    init_info={"init_state":[0.1, 0.2, 0, 0.1, 0]},
    save_render=False,
    legend_list=['INFADP-3000', 'INFADP-190000'],
    use_opt=True,
    opt_args={
        "opt_controller_type": "MPC",
        "num_pred_step": 5, 
        "gamma": 0.99,
        "minimize_options": {
            "max_iter": 200, 
            "tol": 1e-3,
            "acceptable_tol": 1e0,
            "acceptable_iter": 10,
            # "print_level": 5,
        },
        "use_terminal_cost": True,
        # "terminal_cost": terminal_cost,
    }
)

runner.run()