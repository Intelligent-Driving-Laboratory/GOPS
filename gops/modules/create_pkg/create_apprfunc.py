#   Copyright (c) Intelligent Driving Lab(iDLab), Tsinghua University. All Rights Reserved.
#
#  Creator: Hao SUN
#  Description: Create apprfunc
"""

"""

#  Update Date: 2020-12-01, Hao SUN:
import torch.nn as nn


def create_apprfunc(**kwargs):
    # import apprfunc files
    apprfunc_name = kwargs['apprfunc']
    apprfunc_file_name = apprfunc_name.lower()
    try:
        file = __import__(apprfunc_file_name)
    except NotImplementedError:
        raise NotImplementedError('This apprfunc does not exist')


    name = kwargs['name'].upper()
    print(name)
    if hasattr(file,name): #
        apprfunc_cls = getattr(file, name)
        apprfunc = apprfunc_cls(**kwargs)
    else:
        raise NotImplementedError("This apprfunc is not properly defined")

    print("Create apprfunc successfully!"+name)
    return apprfunc

