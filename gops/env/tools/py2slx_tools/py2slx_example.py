#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Export NN_controller to MATLAB/Simulnik
#  Update Date: 2022-10-21, Genjin Xie: Creat py2slx example

# Parameter Description:
"""
   'log_policy_dir_list' is the trained policy loading path
   'trained_policy_iteration_list' is the trained policy corresponding to the number of iteration steps
   'export_controller_name' is the name of the export controller you want
   'save_path' is the absolute save path of the export controller,preferably in the same directory as the
    simulink project files
"""

from py2slx import Py2slxRuner

runer = Py2slxRuner(
    log_policy_dir_list=[r"D:\2_Genjin\THU\Code\gops\results\PPO\221109-211134"],
    trained_policy_iteration_list=["520_opt"],
    export_controller_name=["NN_controller_PPO"],
    save_path=[r"C:\Users\Genjin Xie\Desktop\GOPS_test\vehicle3dof"],
)

runer.py2simulink()
