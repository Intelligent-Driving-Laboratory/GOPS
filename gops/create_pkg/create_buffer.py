#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University

#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Create approximate function module
#  Update Date: 2020-12-13, Hao Sun: add create buffer function


import importlib


def create_buffer(**kwargs):
    trainer = kwargs["trainer"]
    if trainer == "on_serial_trainer" or trainer == "on_sync_trainer":
        buffer = None
    elif (
        trainer == "off_serial_trainer"
        or trainer == "off_async_trainer"
        or trainer == "off_async_trainermix"
        or trainer == "off_sync_trainer"
    ):
        buffer_file_name = kwargs["buffer_name"].lower()
        try:
            module = importlib.import_module("gops.trainer.buffer." + buffer_file_name)
        except NotImplementedError:
            raise NotImplementedError("This buffer does not exist")

        buffer_name = formatter(buffer_file_name)

        if hasattr(module, buffer_name):
            buffer_cls = getattr(module, buffer_name)
            if trainer == "off_serial_trainer":
                buffer = buffer_cls(**kwargs)
            elif (
                trainer == "off_async_trainer"
                or trainer == "off_async_trainermix"
                or trainer == "off_sync_trainer"
            ):
                import ray

                buffer = [
                    ray.remote(num_cpus=1)(buffer_cls).remote(index=idx, **kwargs)
                    for idx in range(kwargs["num_buffers"])
                ]
            else:
                raise NotImplementedError("This trainer is not properly defined")

        else:
            raise NotImplementedError("This buffer is not properly defined")

    print("Create buffer successfully!")
    return buffer


def formatter(src: str, firstUpper: bool = True):
    arr = src.split("_")
    res = ""
    for i in arr:
        res = res + i[0].upper() + i[1:]

    if not firstUpper:
        res = res[0].lower() + res[1:]
    return res
