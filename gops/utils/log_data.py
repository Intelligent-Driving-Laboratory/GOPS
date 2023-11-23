from typing import Sequence, Union


class LogData:
    def __init__(self):
        self.data = {}
        self.counter = {}

    def add_average(self, d: Union[dict, Sequence[dict]]):
        def _add_average(d: dict):
            for k, v in d.items():
                if k not in self.data.keys():
                    self.data[k] = v
                    self.counter[k] = 1
                else:
                    self.data[k] = (self.data[k] * self.counter[k] + v) / (self.counter[k] + 1)
                    self.counter[k] = self.counter[k] + 1

        if isinstance(d, dict):
            _add_average(d)
        elif isinstance(d, Sequence):
            for di in d:
                _add_average(di)
        else:
            raise TypeError(f'Unsupported type {type(d)} for add_average!')

    def pop(self) -> dict:
        data = self.data.copy()
        self.data = {}
        self.counter = {}
        return data
