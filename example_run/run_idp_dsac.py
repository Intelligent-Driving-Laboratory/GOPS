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


from gops.sys_simulator.sys_run import PolicyRunner

runner = PolicyRunner(
    log_policy_dir_list=["../results/DSAC/idpendulum"]+["../results/SAC/idpendulum"],
    trained_policy_iteration_list=["33500_opt"]+["34500_opt"],
    is_init_info=True,
    init_info={"init_state": [-1, 0.05, 0.05, 0, 0.1, 0.1]},
    save_render=False,
    legend_list=["DSAC","SAC"],
    dt=0.01,
    use_opt=False,
)

runner.run()
