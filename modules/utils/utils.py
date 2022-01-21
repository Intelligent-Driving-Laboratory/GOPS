"""


"""
import time

import torch
import torch.nn as nn
import numpy as np

from modules.utils.tensorboard_tools import tb_tags
import random


def get_activation_func(key: str):
    assert isinstance(key, str)

    activation_func = None
    if key == 'relu':
        activation_func = nn.ReLU

    elif key == 'elu':
        activation_func = nn.ELU

    elif key == 'tanh':
        activation_func = nn.Tanh

    elif key == 'linear':
        activation_func = nn.Identity

    if activation_func is None:
        print('input activation name:' + key)
        raise RuntimeError

    return activation_func


def get_apprfunc_dict(key: str, type: str, **kwargs):
    var = dict()
    var['apprfunc'] = kwargs[key + '_func_type']
    var['name'] = kwargs[key + '_func_name']
    var['obs_dim'] = kwargs['obsv_dim']
    var['min_log_std'] = kwargs.get(key + '_min_log_std', float('-inf'))
    var['max_log_std'] = kwargs.get(key + '_max_log_std', float('inf'))

    if type == 'MLP' or type == 'RNN':
        var['hidden_sizes'] = kwargs[key + '_hidden_sizes']
        var['hidden_activation'] = kwargs[key + '_hidden_activation']
        var['output_activation'] = kwargs[key + '_output_activation']
    elif type == 'GAUSS':
        var['num_kernel'] = kwargs[key + '_num_kernel']
    elif type == 'CNN':
        var['hidden_activation'] = kwargs[key + '_hidden_activation']
        var['output_activation'] = kwargs[key + '_output_activation']
        var['conv_type'] = kwargs[key + '_conv_type']
    elif type == 'CNN_SHARED':
        if key == 'feature':
            var['conv_type'] = kwargs['conv_type']
        else:
            var['feature_net'] = kwargs['feature_net']
            var['hidden_activation'] = kwargs[key + '_hidden_activation']
            var['output_activation'] = kwargs[key + '_output_activation']
    elif type == 'POLY':
        pass
    else:
        raise NotImplementedError

    if kwargs['action_type'] == 'continu':
        var['act_high_lim'] = kwargs['action_high_limit']
        var['act_low_lim'] = kwargs['action_low_limit']
        var['act_dim'] = kwargs['action_dim']
    else:
        var['act_num'] = kwargs['action_num']

    return var


def change_type(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):  # add this line
        return obj.tolist()  # add this line
    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = change_type(v)
        return obj
    elif isinstance(obj, list):
        for i, o in enumerate(obj):
            obj[i] = change_type(o)
        return obj
    else:
        return obj


def random_choice_with_index(obj_list):
    obj_len = len(obj_list)
    random_index = random.choice(list(range(obj_len)))
    random_value = obj_list[random_index]
    return random_value, random_index


def array_to_scalar(arrayLike):
    """Convert size-1 array to scalar"""
    return arrayLike if isinstance(arrayLike, (int, float)) else arrayLike.item()


# class Timer(object):
#     def __init__(self, writer, tag=tb_tags['time'], step=None):
#         self.writer = writer
#         self.tag = tag
#         self.step = step
#
#     def __enter__(self):
#         self.start = time.time()
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         # print(time.time() - self.start)
#         self.writer.add_scalar(self.tag, time.time() - self.start, self.step)
