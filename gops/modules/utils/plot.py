#   Copyright (c) Intelligent Driving Lab(iDLab), Tsinghua University. All Rights Reserved.
#
#  Creator: Zhang Yuhang
#  Description: plot_figure, load_tensorboard_file
#  Update Date: 2020-11-10, Hao SUN: renew env para
#  Update Date: 2020-11-13, Hao SUN：add new ddpg demo
#  Update Date: 2020-12-11, Hao SUN：move buffer to trainer
#  Update Date: 2020-12-12, Hao SUN：move create_* files to create_pkg
#  Update Date: 2021-01-01, Hao SUN：change name


#  General Optimal control Problem Solver (GOPS)


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from itertools import cycle

import numpy as np


def self_plot(data,
              fname=None,
              xlabel=None,
              ylabel=None,
              legend=None,
              legend_loc="best",
              color_list=None,
              xlim=None,
              ylim=None,
              xtick=None,
              ytick=None,
              yline=None,
              xline=None,
              ncol=1,
              figsize_scalar=1,
              display=True,
             ):
    """
    plot a single figure containing several curves.
    """
    default_cfg = dict()

    default_cfg['fig_size'] = (8.5, 6.5)
    default_cfg['dpi'] = 300
    default_cfg['pad'] = 0.2

    default_cfg['tick_size'] = 8
    default_cfg['tick_label_font'] = 'Times New Roman'
    default_cfg['legend_font'] = {'family': 'Times New Roman', 'size': '8', 'weight': 'normal'}
    default_cfg['label_font'] = {'family': 'Times New Roman', 'size': '9', 'weight': 'normal'}
    
    
    # pre-process
    assert isinstance(data, (dict, list, tuple))

    if isinstance(data, dict):
        data = [data]
    num_data = len(data)

    fig_size = (default_cfg['fig_size'] * figsize_scalar, default_cfg['fig_size'] * figsize_scalar)
    _, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg['dpi'])


    # color list
    if (color_list is None) or len(color_list) < num_data:
        tableau_colors = cycle(mcolors.TABLEAU_COLORS)
        color_list = [next(tableau_colors) for _ in range(num_data)]

    # plot figure
    for (i, d) in enumerate(data):
        plt.plot(d["x"], d["y"], color=color_list[i])

    # legend
    plt.tick_params(labelsize=default_cfg['tick_size'])
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg['tick_label_font']) for label in labels]

    if legend is not None:
        plt.legend(legend, loc=legend_loc, ncol=ncol, prop=default_cfg['legend_font'])

    #  label
    plt.xlabel(xlabel, default_cfg['label_font'])
    plt.ylabel(ylabel, default_cfg['label_font'])

    if yline is not None:
        plt.axhline(yline, ls=":", c="grey")
    if xline is not None:
        plt.axvline(xline, ls=":", c="grey")

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if xtick is not None:
        plt.xticks(xtick)
    if ytick is not None:
        plt.yticks(ytick)
    plt.tight_layout(pad=default_cfg['pad'])

    if fname is None:
        pass
    else:
        plt.savefig(fname)
    
    if display:
        plt.show()


def read_tensorboard(path, keys):
    """
    input the dir of the tensorboard log
    """
    from tensorboard.backend.event_processing import event_accumulator

    if isinstance(keys, str):
        keys = [keys]
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    print("All available keys in Tensorboard", ea.scalars.Keys())
    valid_key_list = [i for i in keys if i in ea.scalars.Keys()]
    assert len(valid_key_list) != 0, "invalid keys"

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


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def start_tensorboard(logdir, port=6006):
    import os
    import time
    import webbrowser
    import platform
    
    sys_name = platform.system()
    if sys_name == 'Linux':
        cmd_line = "gnome-terminal -- tensorboard --logdir {} --port {}".format(logdir, port)
    elif sys_name == 'Windows':
        cmd_line = '''start cmd.exe /k "tensorboard --logdir {} --port {}"'''.format(logdir, port)
    else:
        print("Unsupported os")
        
    os.system(cmd_line)
    time.sleep(3)

    webbrowser.open("http://localhost:{}/".format(port))


if __name__ == '__main__':
    import numpy as np

    x = np.linspace(0, 100)
    y1 = np.power(x, 2)
    y2 = np.power(x, 3)
    y3 = np.power(x, 4)

    data1 = {'x': x, 'y': y1}
    data2 = {'x': x, 'y': y2}
    data3 = {'x': x, 'y': y3}
    data = [data1, data2, data3]

    self_plot(data, 
            './test.tiff',
            xlabel='Xlabel',
            ylabel='Ylabel',
            legend=[r'$\alpha$=  5e-4', r'$\alpha$=  7e-4', r'$\alpha$=10e-4'],
            legend_loc='lower right',
            display=True)

