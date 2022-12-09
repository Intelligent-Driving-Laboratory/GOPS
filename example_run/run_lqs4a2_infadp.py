#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: run a closed-loop system
#  Update: 2022-12-05, Congsheng Zhang: create file


from gops.utils.common_utils import get_args_from_json
from gops.sys_simulator.sys_run import PolicyRunner
import torch
from gops.algorithm.infadp import ApproxContainer
import os
import argparse

# Load arguments of approximate function
def load_args(log_policy_dir):
    json_path = os.path.join(log_policy_dir, "config.json")
    parser = argparse.ArgumentParser()
    args_dict = vars(parser.parse_args())
    args = get_args_from_json(json_path, args_dict)
    return args


def load_apprfunc(log_policy_dir, trained_policy_iteration):
    # Create apprfunc
    args = load_args(log_policy_dir)
    networks = ApproxContainer(**args)

    # Load trained apprfunc
    log_path = log_policy_dir + "/apprfunc/apprfunc_{}.pkl".format(trained_policy_iteration)
    networks.load_state_dict(torch.load(log_path))
    return networks


# Load value approximate function
value_net = load_apprfunc("../results/INFADP/lqs4a2", "115000_opt").v


# Define terminal cost of MPC controller
def terminal_cost(obs):
    obs = obs.unsqueeze(0)
    return -value_net(obs).squeeze(-1)


runner = PolicyRunner(
    log_policy_dir_list=["../results/INFADP/lqs4a2"] * 1,
    trained_policy_iteration_list=["115000_opt"],
    is_init_info=True,
    init_info={"init_state": [0.5, 0.2, 0.5, 0.1]},
    save_render=False,
    legend_list=["INFADP-115000"],
    use_opt=True,
    opt_args={
        "opt_controller_type": "MPC",
        "num_pred_step": 5,
        "gamma": 0.99,
        "minimize_options": {
            "max_iter": 200,
            "tol": 1e-4,
            "acceptable_tol": 1e-2,
            "acceptable_iter": 10,
        },
        "use_terminal_cost": True,
        "terminal_cost": terminal_cost,
    },
)

runner.run()
