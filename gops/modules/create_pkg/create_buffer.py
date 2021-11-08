#   Copyright (c) Intelligent Driving Lab(iDLab), Tsinghua University. All Rights Reserved.
#
#  Creator: Hao SUN
#  Description: Create buffer
"""

"""
#  Update Date: 2020-12-13, Hao SUN: add create buffer function
import ray
from ..trainer.buffer.replay_buffer import ReplayBuffer


def create_buffer(**kwargs):
    trainer = kwargs['trainer']
    if trainer == 'on_serial_trainer' or trainer == 'on_sync_trainer':
        buffer = None
    elif trainer == 'off_serial_trainer' or trainer == 'off_async_trainer' or trainer == 'off_async_trainermix':
        buffer_file_name = kwargs['buffer_name'].lower()
        try:
            file = __import__(buffer_file_name)
        except NotImplementedError:
            raise NotImplementedError('This buffer does not exist')

        buffer_name = formatter(buffer_file_name)

        if hasattr(file, buffer_name):  #
            buffer_cls = getattr(file, buffer_name)  # 返回
            if trainer == 'off_serial_trainer':
                buffer = buffer_cls(**kwargs)
            elif trainer == 'off_async_trainer' or trainer == 'off_async_trainermix':
                buffer = [ray.remote(num_cpus=1)(ReplayBuffer).remote(**kwargs) for _ in range(kwargs['num_buffers'])]
            else:
                raise NotImplementedError("This trainer is not properly defined")

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
