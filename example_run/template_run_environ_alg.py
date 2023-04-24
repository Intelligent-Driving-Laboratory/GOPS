#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: template for running policy by PolicyRunner
#  Update: 2022-12-11, Zhilong Zheng: create example template

from gops.sys_simulator.call_terminal_cost import load_apprfunc
from gops.sys_simulator.sys_run import PolicyRunner

# Load value approximate function
value_net = load_apprfunc("../results/INFADP/lqs4a2_poly", "115000_opt").v

# Define terminal cost of MPC controller
def terminal_cost(obs):
    obs = obs.unsqueeze(0)
    return -value_net(obs).squeeze(-1)

runner = PolicyRunner(
    # Parameters for policies to be run
    log_policy_dir_list=["../results/INFADP/lqs4a2_mlp",
                         "../results/INFADP/lqs4a2_mlp",
                         "../results/INFADP/lqs4a2_mlp",
                         "../results/INFADP/lqs4a2_poly"],
    trained_policy_iteration_list=["4000",
                                   "5000",
                                   "6000",
                                   "115000_opt"],
    
    # Save environment animation or not
    save_render=False,
    
    # Customize plot range
    # Option 1: no plot range
    plot_range=None,
    # Option 2: plot time steps in [a, b]
    plot_range=[0, 100],

    # Legends for each policy in figures
    legend_list=["InfADP-4000-mlp",
                 "InfADP-5000-mlp",
                 "InfADP-6000-mlp",
                 "InfADP-115000-poly"],
    
    # Constrained environment or not
    constrained_env=False,

    # Tracking problem or not
    is_tracking=False,

    # Use adversarial action or not
    use_dist=False,

    # Parameter for time interval between steps
    # Option 1: use Time step for x-axis
    dt=None,
    # Option 2: use Time (s) for x-axis
    dt=0.1,

    # Parameters for environment initial info
    # Option 1: no specified initial info
    is_init_info=False,
    # Option 2: specify initial state
    is_init_info=True,
    init_info={"init_state": [0.5, 0.2, 0.5, 0.1]},

    # Parameters for optimal controller
    # Option 1: no optimal controller
    use_opt=False,
    # Option 2: default optimal controller
    use_opt=True,
    opt_args={"opt_controller_type": "OPT"},
    # Option 3: MPC controller
    use_opt=True,
    opt_args={
        "opt_controller_type": "MPC",
        "num_pred_step": 5,
        "ctrl_interval": 1,
        "gamma": 0.99,
        "minimize_Options": {"max_iter": 200, "tol": 1e-4,
                             "acceptable_tol": 1e-2,
                             "acceptable_iter": 10,},
        # Option 3.1: w/o terminal cost
        "use_terminal_cost": False,
        # Option 3.2: w/ default terminal cost
        "use_terminal_cost": True,
        "terminal_cost": None,
        # Option 3.2: w/ user-defined terminal cost
        "use_terminal_cost": True,
        "terminal_cost": terminal_cost,

        "verbose": 0,
        "mode": "collocation",
    },

    # Parameter for obs noise
    # Option 1: no obs noise
    obs_noise_type=None, 
    obs_noise_data=None,
    # Option 2: normal obs noise
    obs_noise_type="normal",
    obs_noise_data=[[0.] * 4, [0.1] * 4],
    # Option 3: uniform obs noise
    obs_noise_type="uniform",
    obs_noise_data=[[-0.1] * 4, [0.1] * 4],

    # Parameter for action noise
    # Option 1: no action noise
    action_noise_type=None, 
    action_noise_data=None,
    # Option 2: normal action noise
    action_noise_type="normal",
    action_noise_data=[[0.] * 2, [0.1] * 2],
    # Option 3: uniform action noise
    action_noise_type="uniform",
    action_noise_data=[[-0.1] * 2, [0.1] * 2],
)

runner.run()