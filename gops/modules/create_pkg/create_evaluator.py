#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yang GUAN
#  Description: Create evaluator

def create_evaluator( **kwargs):
    evaluator_file_name = kwargs['evaluator_name'].lower()
    try:
        file = __import__(evaluator_file_name)
    except NotImplementedError:
        raise NotImplementedError('This evaluator does not exist')
    evaluator_name = formatter(evaluator_file_name)
    #
    if hasattr(file, evaluator_name): #
        evaluator_cls = getattr(file, evaluator_name)
        evaluator = evaluator_cls(**kwargs)
    else:
        raise NotImplementedError("This evaluator is not properly defined")

    print("Create evaluator successfully!")
    return evaluator

def formatter(src: str, firstUpper: bool = True):
    arr = src.split('_')
    res = ''
    for i in arr:
        res = res + i[0].upper() + i[1:]

    if not firstUpper:
        res = res[0].lower() + res[1:]
    return res