# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""Биномиальная куча.

# Инициализация
heap = BinomialHeap(values=None, comp=lt)   # Создание кучи из элементов value с использованием функции сравнения comp.
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
Все значения, содержащиеся в куче, внутренне хранятся в виде узлов BinomailNode.

@dataclass
class BinomialNode:
    value: Any
    degree: int
    parent: Union["BinomialNode", None]
    child: Union["BinomialNode", None]
    sibling: Union["BinomialNode", None]

----------------------------------------------------------------
 № | Базовые поддерживаемые операции           |   Сложность   |
---|-------------------------------------------|---------------|
 1 | Получение верхнего                        | O(log(N))     |
   |(самого приоритетного) элемента            |               |
----------------------------------------------------------------
 2 | Вставка                                   | O(log(N))     |
----------------------------------------------------------------
 3 | Извлечение верхнего элемента              | O(log(N))     |
----------------------------------------------------------------
 4 | Слияние двух биномиальных куч             | O(log(N))     |
----------------------------------------------------------------
 5 | Удаление произвольного элемента           | O(log(N))     |
----------------------------------------------------------------
 6 | Построение binomial heap                  | O(N)          |
----------------------------------------------------------------
Алгоритмы и детали реализации в коде.
"""

__all__ = ["BinomialHeap", "MinBinomialHeap", "MaxBinomialHeap"]


from dataclasses import dataclass
from operator import lt, gt
from typing import Any, Union, Callable, Tuple
from collections.abc import Iterable
from collections import deque

from alheapy._heap import Heap, HeapIndexError


@dataclass
class BinomialNode:
    """Узел, хранящий один элемент биномиальной кучи.

    :attr value: данные, хранящиеся в узле
    :attr degree: степень узла (количество потомков)
    :attr parent: родительская вершина
    :attr child: левый потомок
    :attr sibling: правый брат

    """
    value: Any
    degree: int
    parent: Union["BinomialNode", None]
    child: Union["BinomialNode", None]
    sibling: Union["BinomialNode", None]


class BinomialHeap(Heap):
    """Биномиальная куча.

    # Все элементы кучи внутри хранятся в виде узлов BinomialNode

    # Публичные атрибуты и @property:
    heap.head - верхний (самый приоритетный элемент)
    heap.items - итератор по элементам кучи

    # Публичные методы:
    heap.push(item) - добавление нового элемента
    heap.pop() - извлечь верхний элемент
    heap.replace(item) - удалить верхний элемент и затем вставить новый
    heap.pushpop(item) - эквивалентно последовательному вызову heap.push() и heap.pop(),
        но использует лишь одну операцию просевивания и не изменяет структуры узлов, что удваивает производительность
    heap.find(val) - с учетом val, найти узел с таким значением
    heap.update(old, new) - заменить первое вхождение элемента new значением old
    heap.remove(val) - удалить первый найденный элемент со значением val
    heap.merge(*args) - слиение кучи с элементами из *args
    heap.comp(a, b) - True если приоритет a больше (или равен), чем у b и False в противном случа

    """
    def __init__(self, values: Iterable = None, comp: Callable[[Any, Any], bool] = lt):
        self._compfunc = comp
        self._roots: deque[BinomialNode] = deque()
        self.size = 0   # Кол-во элементов в куче
        if values:
            self.merge(values)

    def _root(self) -> Tuple[Union[None, BinomialNode], int]:
        """Поиск корня с самым приоритетным значением"""
        if self.size == 0:
            return None, -1

        top_priority = self._roots[0]
        pos = 0
        for i, root in enumerate(self._roots):
            if self._compn(root, top_priority):
                top_priority = root
                pos = i

        return top_priority, pos

    @property
    def head(self) -> Any:
        """Верхний элемент"""
        root, _ = self._root()
        return root.value if root else None

    def _nodes(self) -> Iterable[BinomialNode]:
        """Обход по узлам кучи"""
        def traverse(node: BinomialNode):
            while node is not None:
                yield node
                for v in traverse(node.child):
                    yield v
                node = node.sibling

        for root in self._roots:
            for i in traverse(root):
                yield i

    @property
    def items(self) -> Iterable[Any]:
        """Обход по элементам кучи"""
        for node in self._nodes():
            yield node.value

    def comp(self, first: Any, second: Any) -> bool:
        """True если приоритет first больше (или равен) приоритета second, и False иначе"""
        return self._compfunc(first, second)

    def _compn(self, first: BinomialNode, second: BinomialNode) -> bool:
        """True если приоритет first больше (или равен) приоритета second, и False иначе"""
        return self._compfunc(first.value, second.value)

    def _merge_trees(self, trees: deque[BinomialNode]) -> deque[BinomialNode]:
        """Слияние всех пар деревьев с равными степенями вершнин, на выходе - новый список корней

        Время работы: O(log(N)), где N - суммарное кол-во элементов во всех деревьях,
        кол-во деревьев равно log(N).

        """
        def merge_two_trees(cur: BinomialNode, sib: BinomialNode) -> BinomialNode:
            """Объединяем два дерева с равными степенями вершин"""
            # Дерево с менее приоритетным корнем становится левым ребёнком второго дерева
            if self._compn(sib, cur):
                cur, sib = sib, cur

            cur.child, sib.sibling = sib, cur.child  # Второе дерево становится левым сыном первого
            cur.child.parent = cur
            cur.degree += 1
            return cur

        if not trees:
            return self._roots

        # Упорядочиваем все деревья в порядке невозрастания степеней, за линейное по количеству узлов время.
        rootlist = deque()
        heap = self._roots
        while trees and heap:
            if trees[0].degree > heap[0].degree:
                rootlist.append(trees.popleft())
            else:
                rootlist.append(heap.popleft())
        rootlist.extend(max(trees, heap))

        # Образуем кучу heap из очереди деревьев rootlist
        # В heap узлы будут располагаться в порядке убывания степеней
        heap = deque([rootlist.popleft()])
        while rootlist:
            if heap and heap[-1].degree == rootlist[0].degree:
                rootlist.appendleft(merge_two_trees(heap.pop(), rootlist.popleft()))
            else:
                heap.append(rootlist.popleft())

        return heap

    def _merge(self, *args: Iterable) -> deque[BinomialNode]:
        """Объединение элементов из *args и текущей кучи в новую кучу"""
        trees = deque()
        for arg in args:
            if not arg:
                continue

            if isinstance(arg, BinomialHeap):
                # Объединяем списки корней в порядке неубывания степеней
                trees.extend(arg._roots)
                self.size += arg.size
            else:
                # Создаём len(arg) очередь из одноэлементных биномиальных деревьев
                items = deque(BinomialNode(val, 0, None, None, None) for val in arg)
                self.size += len(items)
                trees.extend(items)

        # Сливаем все пары с одинаковыми степенями корней.
        return self._merge_trees(trees)

    def merge(self, *args: Iterable):
        self._roots = self._merge(*args)

    def push(self, item: Any):
        """Добавление элемента в кучу"""
        self.size += 1
        node = BinomialNode(item, 0, None, None, None)
        self._roots = self._merge_trees(deque([node]))

    def pop(self) -> Any:
        """Удалить и вернуть верхний элемент"""
        if self.size == 0:
            raise HeapIndexError("pop from empty heap")

        top, pos = self._root()   # Корень с наибольшим приоритетом
        self._roots = deque(self._roots[i] for i in range(len(self._roots)) if i != pos)   # Удаляем этот корень из кучи
        self.size -= 1

        # Находим всех детей удаляемого корня и обновляем их поля parent и sibling
        children = deque()
        child = top.child
        while child is not None:
            children.append(child)
            child.parent = None
            sibling = child.sibling
            child.sibling = None
            child = sibling

        self._roots = self._merge_trees(children)   # Слияние кучи с потомками вырезанного корня
        return top.value

    def replace(self, item: Any) -> Any:
        """Удалить и вернуть верхний элемент, а потом вставить новый

        Среднее время работы составляет O(log(N)), а в худшем - O(log^2(N)) из-за просеивания вниз.
        """
        if self.size == 0:
            raise HeapIndexError("pop from empty heap")
        top, _ = self._root()

        # Заменяем значение в корневом узле и восстанваливаем свойство кучи просеиванием вниз
        to_return = top.value
        top.value = item
        self._siftdown(top)
        return to_return

    def pushpop(self, item: Any) -> Any:
        """Эквивалент последовательному вызову push() и pop(), выполняющий меньше операций

        Среднее время работы составляет O(log(N)), а в худшем - O(log^2(N)) из-за просеивания вниз.

        """
        top, _ = self._root()

        # Если куча пустая или добавляемый элемент стал бы верхним,
        # достаточно просто вернуть его, не внося лишний раз изменение в кучу
        if top is None or self.comp(item, top.value):
            return item

        # Заменяем значение в корневом узле и восстанваливаем свойство кучи просеиванием вниз
        to_return = top.value
        top.value = item
        self._siftdown(top)
        return to_return

    def find(self, val: Any) -> Union[None, BinomialNode]:
        """Поиск узла, хранящего значение val.

        Время работы составляет O(N), т.к. выполняется обход по куче.
        Алгоритм можно улучшить, отсекая деревья, корень которых имеет меньший приоритет,
        чем у искомого элемента.

        """
        for node in self._nodes():
            if node.value == val:
                return node

    def update(self, old: Any, new: Any):
        """Заменить первый найденный элемент old элементом new"""
        node = self.find(old)
        if node is not None:
            node.value = new
            self._siftup(node)
            self._siftdown(node)

    def remove(self, val: Any):
        """Удаление из кучи первого найденного элемента со значением val

        Алгоритм:
        1) находим узел с нужным значением
        2) Находим верхний элемент, присваиваем его значение найденному в шаге 1 узлу
        3) Просеиваем значение найденного в шаге 1 узла вверх
        4) Вызываем pop()
        Время работы O(log(N))

        """
        node = self.find(val)
        if node is not None:
            top, _ = self._root()
            node.value = top.value
            self._siftup(node)
            self.pop()

    def _siftup(self, node: BinomialNode):
        """Просеивание вверх
        Перемещаются только значения в узлах, структура дерева остается прежней.
        Время работы O(log(N)).
        """
        while node.parent is not None:
            if self._compn(node.parent, node):
                break   # Порядок кучи восстановлен
            node.value, node.parent.value = node.parent.value, node.value
            node = node.parent

    def _siftdown(self, node: BinomialNode):
        """Просеивание вниз
        Перемещаются только значения в узлах, структура дерева остается прежней.
        Время работы O(log^2(N)) в худшем случае.
        """
        while node.child is not None:
            child = node.child
            top_priority = child
            child = child.sibling
            while child is not None:
                if self._compn(child, top_priority):
                    top_priority = child
                child = child.sibling

            if self._compn(node, top_priority):
                break   # Порядок кучи восстановлен
            node.value, top_priority.value = top_priority.value, node.value
            node = top_priority

    def __add__(self, other: Iterable) -> "BinomialHeap":
        new = BinomialHeap(other)
        new.merge(self)
        return new

    def __iadd__(self, other: Iterable) -> "BinomialHeap":
        self.merge(other)
        return self

    def __str__(self):
        return "BinomialHeap(size={}, head={})".format(self.size, self.head)

    def __bool__(self):
        return self.size > 0

    def __len__(self):
        return self.size

    def __contains__(self, item: Any):
        return self.find(item) is not None

    def __iter__(self):
        return self.items


class MinBinomialHeap(BinomialHeap):
    def __init__(self, values=None):
        super(MinBinomialHeap, self).__init__(values=values, comp=lt)

    def __str__(self):
        return "MinBinomialHeap(size={}, head={})".format(self.size, self.head)


class MaxBinomialHeap(BinomialHeap):
    def __init__(self, values=None):
        super(MaxBinomialHeap, self).__init__(values=values, comp=gt)

    def __str__(self):
        return "MaxBinomialHeap(size={}, head={})".format(self.size, self.head)


def main():
    heap = BinomialHeap()
    heap.merge([1, 2, 3], BinomialHeap([4, 5]))   # В кучу добавятся элементы 1-5
    print(heap.head)                              # Напечатает 1 - верхний элемент кучи
    heap.push(-5)                                 # Добавление в кучу элемента -5
    print(heap.head)                              # Напечатает -5
    top = heap.pop()                              # Взятие верхнего элемента из кучи, top = -5
    print(top)
    top = heap.replace(6)                         # Замена верхнего элемента значением 6, top = 1
    print(top)
    top = heap.pushpop(7)                         # Эквивалент последовательному вызову push() и pop(), top = 2
    print(top)


if __name__ == "__main__":
    main()
