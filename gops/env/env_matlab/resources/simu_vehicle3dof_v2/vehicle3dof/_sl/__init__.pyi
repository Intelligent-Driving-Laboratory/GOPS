"""Binding for simulink types, for int values, most have uint8 data type."""
from __future__ import annotations
import vehicle3dof._sl
import typing

__all__ = [
    "BigEndianIEEEDouble",
    "IEEESingle",
    "LittleEndianIEEEDouble",
    "rtGetInf",
    "rtGetInfF",
    "rtGetMinusInf",
    "rtGetMinusInfF",
    "rtInf",
    "rtInfF",
    "rtIsInf",
    "rtIsInfF",
    "rtIsNaN",
    "rtIsNaNF",
    "rtMinusInf",
    "rtMinusInfF",
    "rtNaN",
    "rtNaNF",
    "rt_InitInfAndNaN",
]

class BigEndianIEEEDouble:
    class BigEndianIEEEDouble_words_T:
        @property
        def wordH(self) -> int:
            """
            :type: int
            """
        @wordH.setter
        def wordH(self, arg0: int) -> None:
            pass
        @property
        def wordL(self) -> int:
            """
            :type: int
            """
        @wordL.setter
        def wordL(self, arg0: int) -> None:
            pass
        pass
    @property
    def words(self) -> BigEndianIEEEDouble.BigEndianIEEEDouble_words_T:
        """
        :type: BigEndianIEEEDouble.BigEndianIEEEDouble_words_T
        """
    @words.setter
    def words(self, arg0: BigEndianIEEEDouble.BigEndianIEEEDouble_words_T) -> None:
        pass
    pass

class IEEESingle:
    class IEEESingle_words_T:
        @property
        def wordLreal(self) -> float:
            """
            :type: float
            """
        @wordLreal.setter
        def wordLreal(self, arg0: float) -> None:
            pass
        @property
        def wordLuint(self) -> int:
            """
            :type: int
            """
        @wordLuint.setter
        def wordLuint(self, arg0: int) -> None:
            pass
        pass
    @property
    def words(self) -> IEEESingle.IEEESingle_words_T:
        """
        :type: IEEESingle.IEEESingle_words_T
        """
    @words.setter
    def words(self, arg0: IEEESingle.IEEESingle_words_T) -> None:
        pass
    pass

class LittleEndianIEEEDouble:
    class LittleEndianIEEEDouble_words_T:
        @property
        def wordH(self) -> int:
            """
            :type: int
            """
        @wordH.setter
        def wordH(self, arg0: int) -> None:
            pass
        @property
        def wordL(self) -> int:
            """
            :type: int
            """
        @wordL.setter
        def wordL(self, arg0: int) -> None:
            pass
        pass
    @property
    def words(self) -> LittleEndianIEEEDouble.LittleEndianIEEEDouble_words_T:
        """
        :type: LittleEndianIEEEDouble.LittleEndianIEEEDouble_words_T
        """
    @words.setter
    def words(
        self, arg0: LittleEndianIEEEDouble.LittleEndianIEEEDouble_words_T
    ) -> None:
        pass
    pass

def rtGetInf() -> float:
    pass

def rtGetInfF() -> float:
    pass

def rtGetMinusInf() -> float:
    pass

def rtGetMinusInfF() -> float:
    pass

def rtIsInf(value: float) -> bool:
    pass

def rtIsInfF(value: float) -> bool:
    pass

def rtIsNaN(value: float) -> bool:
    pass

def rtIsNaNF(value: float) -> bool:
    pass

def rt_InitInfAndNaN(realSize: int) -> None:
    pass

rtInf = 0.0
rtInfF = 0.0
rtMinusInf = 0.0
rtMinusInfF = 0.0
rtNaN = 0.0
rtNaNF = 0.0
