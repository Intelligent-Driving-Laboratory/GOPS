#   Copyright (c) Intelligent Driving Lab(iDLab), Tsinghua University. All Rights Reserved.
#
#  Creator: Yang GUAN
#  Description: Create sampler

def create_sampler(**kwargs):
    sampler_file_name = kwargs['sampler_name'].lower()
    try:
        file = __import__(sampler_file_name)
        print(file)
    except NotImplementedError:
        raise NotImplementedError('This sampler does not exist')
    sampler_name = formatter(sampler_file_name)
    #
    if hasattr(file, sampler_name): #
        sampler_cls = getattr(file, sampler_name)
        sampler = sampler_cls(**kwargs)
    else:
        raise NotImplementedError("This sampler is not properly defined")

    print("Create sampler successfully!")
    return sampler

def formatter(src: str, firstUpper: bool = True):
    arr = src.split('_')
    res = ''
    for i in arr:
        res = res + i[0].upper() + i[1:]

    if not firstUpper:
        res = res[0].lower() + res[1:]
    return res