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


from gops.sys_simulator.call_terminal_cost import load_apprfunc
from gops.sys_simulator.sys_run import PolicyRunner

# Load value approximate function
value_net = load_apprfunc("../results/INFADP/lqs4a2_poly", "115000_opt").v

# Define terminal cost of MPC controller
def terminal_cost(obs):
    obs = obs.unsqueeze(0)
    return -value_net(obs).squeeze(-1)

runner = PolicyRunner(
    log_policy_dir_list=["../results/INFADP/lqs4a2_mlp",
                         "../results/INFADP/lqs4a2_mlp",
                         "../results/INFADP/lqs4a2_mlp",
                         "../results/INFADP/lqs4a2_poly"],
    trained_policy_iteration_list=["4000",
                                   "5000",
                                   "6000",
                                   "115000_opt"],
    is_init_info=True,
    init_info={"init_state": [0.5, 0.2, 0.5, 0.1]},
    save_render=False,
    legend_list=["InfADP-4000-mlp",
                 "InfADP-5000-mlp",
                 "InfADP-6000-mlp",
                 "InfADP-115000-poly"],
    use_opt=True,
    opt_args={
        "opt_controller_type": "MPC",
        "num_pred_step": 5,
        "gamma": 0.99,
        "minimize_options": {"max_iter": 200, "tol": 1e-4,
                             "acceptable_tol": 1e-2,
                             "acceptable_iter": 10,},
        "use_terminal_cost": True,
        "terminal_cost": terminal_cost,
    },
    dt=None,  # time interval between steps
)

runner.run()
