# -*- coding: utf-8 -*-
"""Биномиальная куча.

# Инициализация
heap = ThinHeap(values=None, comp=lt)   # Создание кучи из элементов value с использованием функции сравнения comp.
По умолчанию comp=lt, поэтому верхним элементом будет минимальным. В общем случае функция comp принимает
два аргумента - элемента и возвращает True, если приоритет первого больше, чем у второго или False, если наоборот

heap.merge([1, 2, 3], BinomialHeap([4, 5]))   # В кучу добавятся элементы 1-5
heap.top                                      # Напечатает 1 - верхний элемент кучи
heap.push(-5)                                 # Добавление в кучу элемента -5
heap.head                                     # Напечатает -5
top = heap.pop()                              # Взятие верхнего элемента из кучи, top = -5
top = heap.replace(6)                         # Замена верхнего элемента значением 6, top = 1
top = pushpop(7)                              # Эквивалент последовательному вызову push() и pop(), top = 2
heap.remove(3)                                # Удаление первого найденного элемента со значением 3
heap.items                                    # Итератор по элементам кучи
# Важная деталь реализации:
Все значения, содержащиеся в куче, внутренне хранятся в виде узлов ThinNode.

@dataclass
class ThinNode:
    value: Any
    rank: int
    child: Union["ThinNode", None]
    left: Union["ThinNode", None]
    right: Union["ThinNode", None]

----------------------------------------------------------------
 № | Базовые поддерживаемые операции           |   Сложность   |
---|-------------------------------------------|---------------|
 1 | Получение верхнего                        | O(1)          |
   |(самого приоритетного) элемента            |               |
----------------------------------------------------------------
 2 | Вставка                                   | O(1)          |
----------------------------------------------------------------
 3 | Извлечение верхнего элемента              | O(log(N))     |
----------------------------------------------------------------
 4 | Слияние двух токних куч                   | O(1)          |
----------------------------------------------------------------
 5 | Построение thin heap                      | O(N)          |
----------------------------------------------------------------
Алгоритмы и детали реализации в коде.
"""

__all__ = ["ThinHeap", "MinThinHeap", "MaxThinHeap"]


from dataclasses import dataclass
from operator import lt, gt
from typing import Any, Union, Callable
from collections.abc import Iterable

from alheapy._heap import Heap, HeapIndexError


@dataclass
class ThinNode:
    """Узел, хранящий один элемент биномиальной кучи.

    :attr value: данные, хранящиеся в узле
    :attr degree: степень узла (количество потомков)
    :attr parent: родительская вершина
    :attr child: левый потомок
    :attr left: левый брат ил родитель, если это самй левый узел на уровне
    :attr right: указатель на правого брата

    """
    value: Any
    rank: int
    child: Union["ThinNode", None]
    left: Union["ThinNode", None]
    right: Union["ThinNode", None]


class ThinHeap(Heap):
    """Тонкая куча.

    # Все элементы кучи внутри хранятся в виде узлов ThinNode

    # Публичные атрибуты и @property:
    self.size - размер кучи
    heap.head - верхний (самый приоритетный элемент)
    heap.items - итератор по элементам кучи

    # Публичные методы:
    heap.push(item) - добавление нового элемента
    heap.pop() - извлечь верхний элемент
    heap.replace(item) - удалить верхний элемент и затем вставить новый
    heap.pushpop(item) - эквивалентно последовательному вызову heap.push() и heap.pop(),
        но использует лишь одну операцию просевивания и не изменяет структуры узлов, что удваивает производительность
    heap.find(val) - с учетом val, найти узел с таким значением
    heap.merge(*args) - слиение кучи с элементами из *args
    heap.comp(a, b) - True если приоритет a больше (или равен), чем у b и False в противном случа

    # Статичсекие методы:
    ThinHeap.is_thin(node) - проверка узла на тонкость

    """
    def __init__(self, values: Iterable = None, comp: Callable[[Any, Any], bool] = lt):
        self._compfunc = comp
        self._head: Union[ThinNode, None] = None
        self._tail: Union[ThinNode, None] = None
        self.size = 0   # Кол-во элементов в куче
        if values:
            self.merge(values)

    @staticmethod
    def is_thin(node: ThinNode) -> bool:
        """Проверка узла на тонкость"""
        if node.rank == 1:
            return node.child is None
        else:
            return node.child and node.child.rank + 1 != node.rank

    @property
    def head(self) -> Any:
        """Верхний элемент"""
        return self._head.value if self._head else None

    def _nodes(self) -> Iterable[ThinNode]:
        """Обход по узлам кучи"""
        def traverse(node):
            if node is None:
                return
            yield node

            right = node
            while right is not None:
                yield right
                for v in traverse(right.child):
                    yield v
                right = right.right

        for n in traverse(self._head):
            yield n

    @property
    def items(self) -> Iterable[Any]:
        """Обход по элементам кучи"""
        for node in self._nodes():
            yield node.value

    def comp(self, first: Any, second: Any) -> bool:
        """True если приоритет first больше (или равен) приоритета second, и False иначе"""
        return self._compfunc(first, second)

    def _compn(self, first: ThinNode, second: ThinNode) -> bool:
        """True если приоритет first больше (или равен) приоритета second, и False иначе"""
        return self._compfunc(first.value, second.value)

    def merge(self, *args: Iterable):
        """Слияние кучи с элементами из args"""
        for arg in args:
            if not arg:
                continue

            if isinstance(arg, ThinHeap):
                self.size += arg.size
                if self._compn(arg._head, self._head):
                    arg._tail.right = self._head
                    self._head = arg._head
                else:
                    self._tail.right = arg._head
                    self._tail = arg._tail
            else:
                for item in arg:
                    self.push(item)

    def _insert(self, node: ThinNode):
        """Вставка узла в кучу за O(1)"""
        if self._head is None:
            self._head, self._tail = node, node
        elif self._compn(node, self._head):
            node.right = self._head
            self._head = node
        else:
            self._tail.right = node
            self._tail = node

    def push(self, item: Any):
        """Добавление элемента в кучу за O(1)"""
        self.size += 1
        self._insert(ThinNode(item, 0, None, None, None))

    def pop(self) -> Any:
        """Удалить и вернуть верхний элемент

        Стоимость: O(log N).

        """
        if self.size == 0:
            raise HeapIndexError("pop from empty heap")
        self.size -= 1
        root = self._head
        to_return = root.value

        # Удаляем корень из корневого списка
        self._head = root.right
        if self._head is None:
            self._tail = None

        # Снимаем тонкость с детей удаляемого корня и добавляем их в корневой список
        child = root.child
        while child is not None:
            if ThinHeap.is_thin(child):
                child.rank -= 1
            temp = child.right
            child.left = None
            child.right = None
            self._insert(child)
            child = temp

        if self._head is None:
            return to_return

        # Объединяем все корни одного ранга
        trees = {}   # Корни деревьев кучи по ключам rank
        current = self._head
        while current is not None:
            right = current.right
            while current.rank in trees:
                if self._compn(trees[current.rank], current):
                    current, trees[current.rank] = trees[current.rank], current

                trees[current.rank].right = current.child
                if current.child:
                    current.child.left = trees[current.rank]
                trees[current.rank].left = current
                current.child = trees[current.rank]

                trees.pop(current.rank)
                current.rank += 1

            trees[current.rank] = current
            current = right

        self._head = None
        for r in sorted(trees.keys()):
            trees[r].right = None
            trees[r].left = None
            self._insert(trees[r])

        return to_return

    def replace(self, item: Any) -> Any:
        """Удалить и вернуть верхний элемент, а потом вставить новый

        Стоимость: O(log(N)).
        """
        if self.size == 0:
            raise HeapIndexError("pop from empty heap")

        to_return = self.pop()
        self.push(item)
        return to_return

    def pushpop(self, item: Any) -> Any:
        """Эквивалент последовательному вызову push() и pop(), выполняющий меньше операций

        Стоимость: O(log(N))
        """
        if self.comp(item, self._head.value):
            return item
        self.push(item)
        return self.pop()

    def find(self, val: Any) -> Union[None, ThinNode]:
        """Поиск узла, хранящего значение val.

        Время работы составляет O(N), т.к. выполняется обход по куче.
        Алгоритм можно улучшить, отсекая деревья, корень которых имеет меньший приоритет,
        чем у искомого элемента.

        """
        for node in self._nodes():
            if node.value == val:
                return node

    def __add__(self, other: Iterable) -> "ThinHeap":
        new = ThinHeap(other)
        new.merge(self)
        return new

    def __iadd__(self, other: Iterable) -> "ThinHeap":
        self.merge(other)
        return self

    def __str__(self):
        return "ThinHeap(size={}, head={})".format(self.size, self.head)

    def __bool__(self):
        return self.size > 0

    def __len__(self):
        return self.size

    def __contains__(self, item: Any):
        return self.find(item) is not None

    def __iter__(self):
        return self.items


class MinThinHeap(ThinHeap):
    def __init__(self, values=None):
        super(MinThinHeap, self).__init__(values=values, comp=lt)

    def __str__(self):
        return "MinThinHeap(size={}, head={})".format(self.size, self.head)


class MaxThinHeap(ThinHeap):
    def __init__(self, values=None):
        super(MaxThinHeap, self).__init__(values=values, comp=gt)

    def __str__(self):
        return "MaxThinHeap(size={}, head={})".format(self.size, self.head)


def main():
    pass


if __name__ == "__main__":
    main()
