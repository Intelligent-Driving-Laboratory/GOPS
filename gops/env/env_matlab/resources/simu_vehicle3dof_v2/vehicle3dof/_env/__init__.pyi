"""Define env-related bindings."""
from __future__ import annotations
import vehicle3dof._env
import typing
import numpy

__all__ = [
    "BoxFloat64",
    "Discrete",
    "EnvSpec",
    "IndexingMode",
    "MultiBinary",
    "MultiDiscrete",
]

class BoxFloat64:
    def __contains__(self, arg0: numpy.ndarray) -> bool: ...
    def __eq__(self, arg0: BoxFloat64) -> bool: ...
    def __getstate__(self) -> tuple: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def contains(self, arg0: numpy.ndarray) -> bool: ...
    @property
    def high(self) -> numpy.ndarray:
        """
        :type: numpy.ndarray
        """
    @property
    def low(self) -> numpy.ndarray:
        """
        :type: numpy.ndarray
        """
    @property
    def shape(self) -> tuple:
        """
        :type: tuple
        """
    __hash__ = None
    dtype: numpy.dtype[float64]  # value = dtype('float64')
    pass

class Discrete:
    def __contains__(self, arg0: int) -> bool: ...
    def __eq__(self, arg0: Discrete) -> bool: ...
    def __getstate__(self) -> tuple: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def contains(self, arg0: int) -> bool: ...
    @property
    def n(self) -> int:
        """
        :type: int
        """
    __hash__ = None
    dtype: numpy.dtype[uint64]  # value = dtype('uint64')
    pass

class EnvSpec:
    def __init__(
        self,
        id: str,
        reward_threshold: typing.Optional[float] = None,
        max_episode_steps: typing.Optional[int] = None,
        terminal_bonus_reward: float = 0.0,
        indexing_mode: IndexingMode = IndexingMode.PRESERVE_EMPTY,
        nondeterministic: bool = False,
        auto_reset: bool = False,
        strict_reset: bool = False,
        need_render: bool = False,
        **kwargs,
    ) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def auto_reset(self) -> bool:
        """
        :type: bool
        """
    @property
    def id(self) -> str:
        """
        :type: str
        """
    @property
    def indexing_mode(self) -> IndexingMode:
        """
        :type: IndexingMode
        """
    @property
    def kwargs(self) -> dict:
        """
        :type: dict
        """
    @property
    def max_episode_steps(self) -> typing.Optional[int]:
        """
        :type: typing.Optional[int]
        """
    @property
    def need_render(self) -> bool:
        """
        :type: bool
        """
    @property
    def nondeterministic(self) -> bool:
        """
        :type: bool
        """
    @property
    def reward_threshold(self) -> typing.Optional[float]:
        """
        :type: typing.Optional[float]
        """
    @property
    def strict_reset(self) -> bool:
        """
        :type: bool
        """
    @property
    def terminal_bonus_reward(self) -> float:
        """
        :type: float
        """
    pass

class IndexingMode:
    """
    Members:

      PRESERVE_EMPTY

      PRESERVE_FILL

      COMPRESS
    """

    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    COMPRESS: vehicle3dof._env.IndexingMode  # value = <IndexingMode.COMPRESS: 2>
    PRESERVE_EMPTY: vehicle3dof._env.IndexingMode  # value = <IndexingMode.PRESERVE_EMPTY: 0>
    PRESERVE_FILL: vehicle3dof._env.IndexingMode  # value = <IndexingMode.PRESERVE_FILL: 1>
    __members__: dict  # value = {'PRESERVE_EMPTY': <IndexingMode.PRESERVE_EMPTY: 0>, 'PRESERVE_FILL': <IndexingMode.PRESERVE_FILL: 1>, 'COMPRESS': <IndexingMode.COMPRESS: 2>}
    pass

class MultiBinary:
    def __contains__(self, arg0: typing.List[int]) -> bool: ...
    def __eq__(self, arg0: MultiBinary) -> bool: ...
    def __getstate__(self) -> tuple: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def contains(self, arg0: typing.List[int]) -> bool: ...
    @property
    def n(self) -> int:
        """
        :type: int
        """
    __hash__ = None
    dtype: numpy.dtype[bool_]  # value = dtype('bool')
    pass

class MultiDiscrete:
    def __contains__(self, arg0: typing.List[int]) -> bool: ...
    def __eq__(self, arg0: MultiDiscrete) -> bool: ...
    def __getstate__(self) -> tuple: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def contains(self, arg0: typing.List[int]) -> bool: ...
    @property
    def nvec(self) -> typing.List[int]:
        """
        :type: typing.List[int]
        """
    __hash__ = None
    dtype: numpy.dtype[uint64]  # value = dtype('uint64')
    pass
