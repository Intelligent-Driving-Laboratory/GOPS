#   Copyright (c) Intelligent Driving Lab(iDLab), Tsinghua University. All Rights Reserved.
#
#  Creator: Hao SUN
#  Description: Create algorithm
"""

"""

#  Update Date: 2020-12-01, Hao SUN:


def create_alg(**kwargs):
    alg_name = kwargs['algorithm']
    alg_file_name = alg_name.lower()
    try:
        file = __import__(alg_file_name)
    except NotImplementedError:
        raise NotImplementedError('This algorithm does not exist')

    #
    if hasattr(file, alg_name): #
        alg_cls = getattr(file, alg_name)
        alg = alg_cls(**kwargs)
    else:
        raise NotImplementedError("This algorithm is not properly defined")

    print("Create alg successfully!")
    return alg


# def formatter(src: str, firstUpper: bool = True):
#     arr = src.split('_')
#     res = ''
#     for i in arr:
#         res = res + i[0].upper() + i[1:]
#
#     if not firstUpper:
#         res = res[0].lower() + res[1:]
#     return res