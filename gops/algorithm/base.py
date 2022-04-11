from gops.utils.utils import set_seed

class AlgorithmBase:
    def __init__(self, index, **kwargs):
        set_seed(kwargs["trainer"], kwargs["seed"], index + 300)
