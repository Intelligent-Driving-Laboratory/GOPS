import os

gops_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
algorithm_path = os.path.join(gops_path, "algorithm")
apprfunc_path = os.path.join(gops_path, "apprfunc")
env_path = os.path.join(gops_path, "env")
trainer_path = os.path.join(gops_path, "trainer")
buffer_path = os.path.join(trainer_path, "buffer")
sampler_path = os.path.join(trainer_path, "sampler")


def underline2camel(s: str, first_upper: bool = False) -> str:
    arr = s.split("_")
    if first_upper:
        res = arr.pop(0).upper()
    else:
        res = ""
    for a in arr:
        res = res + a[0].upper() + a[1:]
    return res

def camel2underline(s: str) -> str:
    res = ''
    for i in range(len(s) - 1):
        if s[i].isupper() and s[i+1].islower():
            res = res + "_" + s[i].lower()
        else:
            res = res + s[i].lower()
    res = res + s[-1].lower()
    return res