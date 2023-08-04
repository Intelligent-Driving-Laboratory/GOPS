from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Spec:
    id: str
    entry_point: Callable

    # Environment arguments
    kwargs: dict = field(default_factory=dict)
