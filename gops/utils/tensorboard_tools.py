#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Tensorboard Related Function
#  Update Date: 2021-03-10, Yuhang Zhang: Create codes


import numpy as np
import pandas as pd
import os
import time
import webbrowser
import platform
import signal

import tensorboard.backend.application

DEFAULT_TB_PORT = 6001


def read_tensorboard(path):
    """
    input the dir of the tensorboard log
    """
    import tensorboard
    from tensorboard.backend.event_processing import event_accumulator

    tensorboard.backend.application.logger.setLevel("ERROR")
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    # print("All available keys in Tensorboard", ea.scalars.Keys())
    valid_key_list = ea.scalars.Keys()

    output_dict = dict()
    for key in valid_key_list:
        event_list = ea.scalars.Items(key)
        x, y = [], []
        for e in event_list:
            x.append(e.step)
            y.append(e.value)

        data_dict = {"x": np.array(x), "y": np.array(y)}
        output_dict[key] = data_dict
    return output_dict


def start_tensorboard(logdir, port=DEFAULT_TB_PORT):
    kill_port(port)

    sys_name = platform.system()
    if sys_name == "Linux":
        cmd_line = "gnome-terminal -- tensorboard --logdir {} --port {}".format(
            logdir, port
        )
    elif sys_name == "Windows":
        cmd_line = '''start /b cmd.exe /k "tensorboard --logdir {} --port {}"'''.format(
            logdir, port
        )
    else:
        print("Unsupported os")

    os.system(cmd_line)
    time.sleep(5)

    webbrowser.open("http://localhost:{}/".format(port))


def add_scalars(tb_info, writer, step):
    for key, value in tb_info.items():
        writer.add_scalar(key, value, step)


def get_pids_linux(port):
    with os.popen("lsof -i:{}".format(port)) as res:
        res = res.read().split("\n")
    results = []
    for line in res[1:]:
        if line == "":
            continue
        temp = [i for i in line.split(" ") if i != ""]
        results.append(temp[1])
    return list(set(results))


def get_pids_windows(port):
    with os.popen('netstat -aon|findstr "' + '{}"'.format(port)) as res:
        res = res.read().split("\n")
    results = []
    for line in res:
        temp = [i for i in line.split(" ") if i != ""]
        if len(temp) > 4:
            results.append(temp[4])
    return list(set(results))


def kill_pids_linux(pids):
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGINT)
        except:
            pass


def kill_pid_windows(pids):
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGINT)
        except:
            pass


def kill_port(port=DEFAULT_TB_PORT):
    sys_name = platform.system()
    if sys_name == "Linux":
        pids = get_pids_linux(port)
        kill_pids_linux(pids)
    elif sys_name == "Windows":
        pids = get_pids_windows(port)
        kill_pid_windows(pids)
    else:
        print("Unsupported os")


def save_csv(path, step, value):
    """
    save 2-column-data to csv

    Parameters
    ----------
    path: str
        The target csv file

    step: numpy.array
        The first dim of the data

    value: numpy.array
        The second dim of the data

    Returns
    -------

    """
    df = pd.DataFrame({"Step": step, "Value": value})
    df.to_csv(path, index=False, sep=",")


def save_tb_to_csv(path):
    """
    Parse all tensorboard log file in the give dir (e.g. ./results),
    and save all data as csv

    Parameters
    ----------
    path: str
        Target dir, such as: './results', './runs'
    depth: int, optional
        Layers of the tensorboard log dir

    Returns
    -------
    Non
    """

    data_dict = read_tensorboard(path)
    for data_name in data_dict.keys():
        data_name_format = data_name.replace("\\", "/").replace("/", "_")
        csv_dir = os.path.join(path, "csv")
        os.makedirs(csv_dir, exist_ok=True)
        save_csv(
            os.path.join(csv_dir, "{}.csv".format(data_name_format)),
            step=data_dict[data_name]["x"],
            value=data_dict[data_name]["y"],
        )


tb_tags = {
    "TAR of RL iteration": "Evaluation/1. TAR-RL iteration",
    "TAR of total time": "Evaluation/2. TAR-Total time [s]",
    "TAR of collected samples": "Evaluation/3. TAR-Collected samples",
    "TAR of replay samples": "Evaluation/4. TAR-Replay samples",
    "Buffer RAM of RL iteration": "RAM/RAM-RL iteration",
    "loss_actor": "Loss/loss_actor",
    "loss_critic": "Loss/loss_critic",
    "alg_time": "Time/alg_time",
    "sampler_time": "Time/sampler_time",
    "critic_avg_value": "Train/critic_average_value",
}

if __name__ == "__main__":
    kill_port()
