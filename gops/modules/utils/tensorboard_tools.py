"""
tensorboard related functions

"""
import numpy as np
import os
import time
import webbrowser
import platform
import signal

import tensorboard.backend.application


def read_tensorboard(path):
    """
    input the dir of the tensorboard log
    """
    import tensorboard
    from tensorboard.backend.event_processing import event_accumulator
    tensorboard.backend.application.logger.setLevel('ERROR')
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

        data_dict = {'x': np.array(x), 'y': np.array(y)}
        output_dict[key] = data_dict
    return output_dict


def start_tensorboard(logdir, port=6006):
    kill_port(port)

    sys_name = platform.system()
    if sys_name == 'Linux':
        cmd_line = "gnome-terminal -- tensorboard --logdir {} --port {}".format(logdir, port)
    elif sys_name == 'Windows':
        cmd_line = '''start cmd.exe /k "tensorboard --logdir {} --port {}"'''.format(logdir, port)
    else:
        print("Unsupported os")

    os.system(cmd_line)
    time.sleep(10)

    webbrowser.open("http://localhost:{}/".format(port))


def add_scalars(tb_info, writer, step):
    for key, value in tb_info.items():
        writer.add_scalar(key, value, step)


def get_pids_linux(port):
    with os.popen('lsof -i:{}'.format(port)) as res:
        res = res.read().split('\n')
    results = []
    for line in res[1:]:
        if line == '':
            continue
        temp = [i for i in line.split(' ') if i != '']
        results.append(temp[1])
    return list(set(results))


def get_pids_windows(port):
    with os.popen('netstat -aon|findstr "6006"') as res:
        res = res.read().split('\n')
    results = []
    for line in res:
        temp = [i for i in line.split(' ') if i != '']
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


def kill_port(port=6006):
    sys_name = platform.system()
    if sys_name == 'Linux':
        pids = get_pids_linux(port)
        kill_pids_linux(pids)
    elif sys_name == 'Windows':
        pids = get_pids_windows(port)
        kill_pid_windows(pids)
    else:
        print("Unsupported os")


tb_tags = {'loss_actor': 'Loss/loss_actor',
           'loss_critic': 'Loss/loss_critic',
           'time': 'Performance/time',
           'total_average_return': 'Performance/total_average_return'}


if __name__ == '__main__':
    kill_port()