"""Define env-related bindings."""
from __future__ import annotations
import lqs2a1._env
import typing

__all__ = [
    "ActionRepeatMode",
    "EnvSpec",
    "IndexingMode"
]


class ActionRepeatMode():
    """
    Members:

      SUM_BREAK
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
    SUM_BREAK: lqs2a1._env.ActionRepeatMode # value = <ActionRepeatMode.SUM_BREAK: 0>
    __members__: dict # value = {'SUM_BREAK': <ActionRepeatMode.SUM_BREAK: 0>}
    pass
class EnvSpec():
    def __init__(self, id: str, reward_threshold: typing.Optional[float] = None, max_episode_steps: typing.Optional[int] = None, terminal_bonus_reward: float = 0.0, indexing_mode: IndexingMode = IndexingMode.PRESERVE_EMPTY, nondeterministic: bool = False, auto_reset: bool = False, strict_reset: bool = True, need_render: bool = False, action_repeat: int = 0, action_repeat_mode: ActionRepeatMode = ActionRepeatMode.SUM_BREAK, **kwargs) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def action_repeat(self) -> int:
        """
        :type: int
        """
    @property
    def action_repeat_mode(self) -> ActionRepeatMode:
        """
        :type: ActionRepeatMode
        """
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
class IndexingMode():
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
    COMPRESS: lqs2a1._env.IndexingMode # value = <IndexingMode.COMPRESS: 2>
    PRESERVE_EMPTY: lqs2a1._env.IndexingMode # value = <IndexingMode.PRESERVE_EMPTY: 0>
    PRESERVE_FILL: lqs2a1._env.IndexingMode # value = <IndexingMode.PRESERVE_FILL: 1>
    __members__: dict # value = {'PRESERVE_EMPTY': <IndexingMode.PRESERVE_EMPTY: 0>, 'PRESERVE_FILL': <IndexingMode.PRESERVE_FILL: 1>, 'COMPRESS': <IndexingMode.COMPRESS: 2>}
    pass
