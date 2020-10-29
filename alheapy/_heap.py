# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Union, Any
from collections.abc import Sequence, Iterable


class HeapIndexError(IndexError):
    def __init__(self, message):
        self.message = message


class Heap(ABC):
    """Абстрактный класс кучи, определяющий поведение для операторов сравнения
    __eq__, __ne__, __lt__, __gt__, __le__, __ge__.
    Эти операторы сравнивают приоритетность каждой кучи по верхнему элементу.
    Для этих целей любая куча должна возвращать при обращении self.head верхний элемент и иметь метод self._compn(),
    определяющий аргумент с большим приоритетом. Также кучу можно сравнивать и
    с произвольной коллекцией по первому ее элементу (collection[0]), если же второй аргумент не является кучей и
    не предоставляет доступ к первому элементу, будет возбуждено соответствующее исключение.

    Сравнение куч имеет смысл если обе кучи используют одинаковый метод _compn().
    !!! _compn() может поределять приоритет строго (< >) или нестрого (<= >=),
    поэтому методы __lt__, __gt__, __le__, __ge__ используют дополнительные проверки с вызовами __eq__ или __ne__.
    """

    @property
    @abstractmethod
    def head(self) -> Any:
        """Элемент с наибольшим приоритетом"""
        pass

    @property
    @abstractmethod
    def items(self) -> Iterable:
        """Итератор по элементам кучи"""
        pass

    @abstractmethod
    def comp(self, first: Any, second: Any) -> bool:
        """True если приоритет first больше (или равен) приоритета second, и False иначе"""
        pass

    def __eq__(self, other: Union['Heap', Sequence]) -> bool:
        if isinstance(other, Heap):
            return self.head == other.head
        return self.head == other[0]

    def __ne__(self, other: Union['Heap', Sequence]) -> bool:
        if isinstance(other, Heap):
            return self.head != other.head
        return self.head != other[0]

    def __lt__(self, other: Union['Heap', Sequence]) -> bool:
        if isinstance(other, Heap):
            return self.comp(other.head, self.head) and self != other
        return self.comp(other[0], self.head) and self != other

    def __gt__(self, other: Union['Heap', Sequence]) -> bool:
        if isinstance(other, Heap):
            return self.comp(self.head, other.head) and self != other
        return self.comp(self.head, other[0]) and self != other

    def __le__(self, other: Union['Heap', Sequence]) -> bool:
        if isinstance(other, Heap):
            return self.comp(other.head, self.head) or self == other
        return self.comp(other[0], self.head) or self == other

    def __ge__(self, other: Union['Heap', Sequence]) -> bool:
        if isinstance(other, Heap):
            return self.comp(self.head, other.head) or self == other
        return self.comp(self.head, other[0]) or self == other
