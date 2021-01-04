#   Copyright (c) Intelligent Driving Lab(iDLab), Tsinghua University. All Rights Reserved.
#
#  Creator: Hao SUN
#  Description: Create trainers
"""
resources:

"""

#  Update Date: 2020-12-01, Hao SUN:

#import modules.trainer.serial_trainer

def create_trainer(env,alg,**kwargs):
    trainer_name = kwargs['trainer']
    try:
        file = __import__(trainer_name)
    except NotImplementedError:
        raise NotImplementedError('This trainer does not exist')

    trainer_name_camel = formatter(trainer_name) #
    # get
    if hasattr(file, trainer_name_camel):
        trainer_cls = getattr(file, trainer_name_camel)
        trainer = trainer_cls(alg=alg,env=env,**kwargs)
    else:
        raise NotImplementedError("This trainer is not properly defined")
    print("Create trainer successfully!")
    return trainer


def formatter(src: str, firstUpper: bool = True):
    arr = src.split('_')
    res = ''
    for i in arr:
        res = res + i[0].upper() + i[1:]

    if not firstUpper:
        res = res[0].lower() + res[1:]
    return res