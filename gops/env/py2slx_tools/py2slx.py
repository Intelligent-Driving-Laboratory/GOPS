#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Export trained policy into Simulink for closed-loop validation and check matlab version
#  Update Date: 2022-10-20, Genjin Xie: Creat policy export and matlab check modular

import argparse
import os
import torch
import winreg

from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_sampler import create_sampler
from gops.utils.common_utils import get_args_from_json

from gops.env.py2slx_tools.export import check_jit_compatibility, export_model


class Py2slxRuner:
    """
     GOPS tool for put trained policy back into Simulink for closed-loop validation and check whether user's
     matlab version is correct.

    : param str log_policy_dir_list is the trained policy loading path.
    : param str trained_policy_iteration_list is the trained policy corresponding to the number of iteration steps.
    : param str export_controller_name is the name of the export controller you want.
    : param str save_path is the absolute save path of the export controller,preferably in the same directory
    as the simulink project files.
    """

    def __init__(
        self,
        log_policy_dir_list: list,
        trained_policy_iteration_list: list,
        export_controller_name: list,
        save_path: list,
    ) -> None:
        self.log_policy_dir_list = log_policy_dir_list
        self.trained_policy_iteration_list = trained_policy_iteration_list
        self.export_controller_name = export_controller_name
        self.save_path = save_path
        self.args = None
        self.is_correct = False
        self.matlab_version = None
        self.policy_num = len(self.log_policy_dir_list)
        if self.policy_num != len(self.trained_policy_iteration_list):
            raise RuntimeError(
                "The lenth of policy number is not equal to the number of policy iteration"
            )

        self.args_list = []
        self.algorithm_list = []
        self.__load_all_args()

    @staticmethod
    def __load_args(log_policy_dir: str) -> dict:
        json_path = os.path.join(log_policy_dir, "config.json")
        parser = argparse.ArgumentParser()
        args_dict = vars(parser.parse_args())
        args = get_args_from_json(json_path, args_dict)
        return args

    def __load_all_args(self):
        for i in range(self.policy_num):
            log_policy_dir = self.log_policy_dir_list[i]
            args = self.__load_args(log_policy_dir)
            self.args_list.append(args)
            self.algorithm_list.append(args["algorithm"])

    def __load_env(self):
        env = create_env(**self.args)
        self.args["action_high_limit"] = env.action_space.high
        self.args["action_low_limit"] = env.action_space.low
        return env

    def __load_policy(self, log_policy_dir: str, trained_policy_iteration: str):
        # Create policy
        alg_name = self.args["algorithm"]
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        ApproxContainer = getattr(file, "ApproxContainer")
        networks = ApproxContainer(**self.args)
        print("Create {}-policy successfully!".format(alg_name))

        # Load trained policy
        log_path = log_policy_dir + "/apprfunc/apprfunc_{}.pkl".format(
            trained_policy_iteration
        )
        networks.load_state_dict(torch.load(log_path))
        print("Load {}-policy successfully!".format(alg_name))
        return networks

    def __load_sampler(self,):
        sampler = create_sampler(**self.args)
        return sampler

    # Find matlab version from computer registry
    def __search_matlab_version(self):
        location = r"SOFTWARE\\MathWorks"
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, location)
        countkey = winreg.QueryInfoKey(key)[0]
        matlab_list = ""
        for i in range(countkey):
            try:
                name = winreg.EnumKey(key, i)
                matlab_list = matlab_list + name
            except OSError as error:
                winreg.CloseKey(key)
        return matlab_list

    def __run_data(self):
        for i in range(self.policy_num):
            log_policy_dir = self.log_policy_dir_list[i]
            trained_policy_iteration = self.trained_policy_iteration_list[i]

            self.args = self.args_list[i]
            # env = self.__load_env()
            networks = self.__load_policy(log_policy_dir, trained_policy_iteration)
            sampler = self.__load_sampler()
            model = networks.policy

            example_obs_row = sampler.env.reset()[0]
            # example_obs = torch.from_numpy(sampler.env.reset()).float()
            example_obs = torch.from_numpy(example_obs_row).float()
            save_path = self.save_path[0] + "/{}.pt".format(
                self.export_controller_name[0]
            )
            self.__check_export_model(model, example_obs, save_path)

    def __check_export_model(self, model, example_obs, save_path):
        check_jit_compatibility(model, example_obs)
        export_model(model, example_obs, save_path)

    # Check whether matlab version is correct and open matlab
    def __check_open_matlab(self):
        matlab_version_list = [
            "R2016a",
            "R2016b",
            "R2017a",
            "R2017b",
            "R2018a",
            "R2018b",
            "R2019a",
            "R2019b",
            "R2020a",
            "R2020b",
            "R2021a",
            "R2021b",
            "R2022a",
            "R2022b",
            "R2023a",
            "R2023b",
            "R2024a",
            "R2024b",
            "R2025a",
            "R2025b",
            "R2026a",
            "R2026b",
            "R2027a",
            "R2027b",
        ]
        correct_version_list = [
            "R2021b",
            "R2022a",
            "R2022b",
            "R2023a",
            "R2023b",
            "R2024a",
            "R2024b",
            "R2025a",
            "R2025b",
            "R2026a",
            "R2026b",
            "R2027a",
            "R2027b",
        ]
        connect = ""
        correct_version_str = connect.join(correct_version_list)
        matlab_list = self.__search_matlab_version()
        for i in range(len(matlab_version_list)):
            if matlab_version_list[i] in matlab_list:
                self.matlab_version = matlab_version_list[i]
        if self.matlab_version in correct_version_str:
            self.is_correct = True
        work_dir = self.save_path[0]
        os.chdir(work_dir)
        have_matlab = os.system("matlab")
        if have_matlab != 0:
            print(
                "\033[31mMatlabNotFoundError: Please install MATLAB, and ensure the minimum MATLAB version is "
                "R2021b.\033[0m"
            )
        elif have_matlab == 0 and self.is_correct:
            print("The current MATLAB version is: {}.".format(self.matlab_version))
        elif have_matlab == 0 and (not self.is_correct):
            print(
                "\033[31mMatlabVersionError: The current MATLAB version is {},"
                " please ensure the minimum MATLAB version is R2021b.\033[0m".format(
                    self.matlab_version
                )
            )

    def py2simulink(self):
        self.__run_data()
        self.__check_open_matlab()
