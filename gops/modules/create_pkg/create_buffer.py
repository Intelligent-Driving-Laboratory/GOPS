#   Copyright (c) Intelligent Driving Lab(iDLab), Tsinghua University. All Rights Reserved.
#
#  Creator: Hao SUN
#  Description: Create buffer
"""

"""
#  Update Date: 2020-12-13, Hao SUN: add create buffer function


def create_buffer(**kwargs):
    buffer_file_name = kwargs['buffer_name'].lower()
    try:
        file = __import__(buffer_file_name)
    except NotImplementedError:
        raise NotImplementedError('This buffer does not exist')

    #
    obs_dim = kwargs['obsv_dim']
    act_dim = kwargs['action_dim']
    size = kwargs['buffer_max_size']
    buffer_name = formatter(buffer_file_name)
    #print(buffer_name)

    if hasattr(file, buffer_name): #
        buffer_cls = getattr(file, buffer_name) # 返回
        buffer = buffer_cls(obs_dim = obs_dim,act_dim=act_dim,size=size)
    else:
        raise NotImplementedError("This buffer is not properly defined")

    print("Create buffer successfully!")
    return buffer


def formatter(src: str, firstUpper: bool = True):
    arr = src.split('_')
    res = ''
    for i in arr:
        res = res + i[0].upper() + i[1:]

    if not firstUpper:
        res = res[0].lower() + res[1:]
    return res