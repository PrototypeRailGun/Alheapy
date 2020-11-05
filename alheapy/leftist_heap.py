# -*- coding: utf-8 -*-
"""Левосторонняя куча.

# Инициализация
heap = LeftistHeap(values=None, comp=lt)   # Создание кучи из элементов value с использованием функции сравнения comp.
По умолчанию comp=lt, поэтому верхним элементом будет минимальным. В общем случае функция comp принимает
два аргумента - элемента и возвращает True, если приоритет первого больше, чем у второго или False, если наоборот

heap.merge([1, 2, 3], LeftistHeap([4, 5]))   # В кучу добавятся элементы 1-5
heap.top                                     # Напечатает 1 - верхний элемент кучи
heap.push(-5)                                # Добавление в кучу элемента -5
heap.head                                    # Напечатает -5
top = heap.pop()                             # Взятие верхнего элемента из кучи, top = -5
top = heap.replace(6)                        # Замена верхнего элемента значением 6, top = 1
top = pushpop(7)                             # Эквивалент последовательному вызову push() и pop(), top = 2
heap.remove(3)                               # Удаление первого найденного элемента со значением 3
heap.items                                   # Итератор по элементам кучи, значения перебираются обходом в ширину

# Важная деталь реализации:
Все значения, содержащиеся в куче, внутренне хранятся в виде узлов LeftistNode.

@dataclass
class LeftistNode:
    value: Any                           # Ценное значение (элемент)
    dist: int                            # Расстояние до ближайшего свободного места
    parent: Union["LeftistNode", None]   # Родитель
    left: Union["LeftistNode", None]     # Левый потомок
    right: Union["LeftistNode", None]    # Правый потомок


# Про левостороннюю кучу
Определение:
<Левосторонняя куча> (англ. leftist heap) — двоичное левосторонее дерево (не обязательно сбалансированное),
но с соблюдением порядка кучи (приоитет родителя не меньше приорита любого из потомков).

<Свободной позицией> назовем место в дереве, куда может быть вставлена новая вершина.
Само дерево будет являться свободной позицией, если оно не содержит вершин.
Если же у какой-то внутренней вершины нет сына, то на его месте — свободная позиция.

Условие левосторонней кучи:
Пусть dist(v) — расстояние от вершины v до ближайшей свободной позиции в ее поддереве. У пустых позиций dist=0.
Теперь потребуем для любой вершины v: dist(v.left) >= dist(v.right). Если для какой-то вершины это свойство не
выполняется, то достаточно поменять потомков местами.

Утверждение 1: В двоичном дереве с n вершинами существует свободная позиция на глубине не более log(N).
Утверждение 2: Ближайшая к корню свободная позиция находится в самой правой ветви.

Преимущество левосторонней кучи над двоичной в том, что операция слияние выполняется за O(log(N)) вместо O(N) у бинарной

# Поддерживаемые операции
----------------------------------------------------------------
 № | Базовые поддерживаемые операции           |   Сложность   |
---|-------------------------------------------|---------------|
 1 | Получение верхнего                        | O(1)          |
   |(самого приоритетного) элемента            |               |
----------------------------------------------------------------
 2 | Вставка                                   | O(log(N))     |
----------------------------------------------------------------
 3 | Извлечение верхнего элемента              | O(log(N))     |
----------------------------------------------------------------
 4 | Слияние двух левосторонних куч            | O(log(N))     |
----------------------------------------------------------------
 5 | Удаление произвольного элемента           | O(log(N))     |
----------------------------------------------------------------
 6 | Построение leftist heap                   | O(N)          |
----------------------------------------------------------------

Алгоритмы и детали реализации в коде.
"""

__all__ = ["LeftistHeap", "MinLeftistHeap", "MaxLeftistHeap"]


from dataclasses import dataclass
from copy import deepcopy
from operator import lt, gt
from typing import Any, Tuple, Union, Callable
from collections import deque
from collections.abc import Iterable

from alheapy._heap import Heap, HeapIndexError


@dataclass
class LeftistNode:
    """Узел, хранящий один элемент левосторонней кучи.

    :attr value: данные, хранящиеся в узле
    :attr: dist - расстояние до ближайшего свободного места, 0 если узел имеет 0 или 1 потомков.
    :attr parent: родительская вершина
    :attr left: левый потомок
    :attr right: правый потомок

    """
    value: Any
    dist: int
    parent: Union["LeftistNode", None]
    left: Union["LeftistNode", None]
    right: Union["LeftistNode", None]


class LeftistHeap(Heap):
    """Левосторонняя куча.

    # Все элементы кучи внутри хранятся в виде узлов LeftistNode

    # Публичные атрибуты и @property:
    heap.head - верхний (самый приоритетный элемент)
    heap.items - итератор по элементам кучи

    # Публичные методы:
    heap.push(item) - добавление нового элемента
    heap.pop() - извлечь верхний элемент
    heap.replace(item) - удалить верхний элемент и затем вставить новый
    heap.pushpop(item) - эквивалентно последовательному вызову heap.push() и heap.pop(),
        но использует лишь одну операцию просевивания, что повышает эффективность почти в 2 раза
    heap.find(val) - с учетом val, найти узел с таким значением
    heap.update(old, new) - заменить первое вхождение элемента new значением old
    heap.remove(val) - удалить первый найденный элемент со значением val
    heap.merge(*args) - слиение кучи с элементами из *args
    heap.comp(a, b) - True если приоритет a больше (или равен), чем у b и False в противном случае

    # Статические методы
    LeftistHeap.merge_nodes(first, second, comp) - слияние двух узлов с использованием сравнения приоритетов comp
    LeftistHeap.merge_iterable(*args) - построение кучи из Iterable за O(N)

    """
    def __init__(self, values: Iterable = None, comp: Callable[[Any, Any], bool] = lt):
        self._compfunc = comp
        self._head: Union[LeftistNode, None] = None   # Верхний элемент
        self.size = 0   # Кол-во элементов в куче
        if values:
            self.merge(values)

    @property
    def head(self) -> Any:
        """Верхний элемент"""
        if self._head:
            return self._head.value

    @property
    def items(self) -> Iterable:
        """Итератор обхода в ширину по элементам кучи"""
        if self._head is None:
            return

        queue = deque([self._head])
        while queue:
            node = queue.popleft()

            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)

            yield node.value

    def comp(self, first: Any, second: Any) -> bool:
        """True если приоритет first больше (или равен) приоритета second, и False иначе"""
        return self._compfunc(first, second)

    def _compn(self, first: LeftistNode, second: LeftistNode) -> bool:
        """True если приоритет first больше (или равен) приоритета second, и False иначе"""
        return self._compfunc(first.value, second.value)

    @staticmethod
    def merge_nodes(first: LeftistNode, second: LeftistNode,
                    comp: Callable[[LeftistNode, LeftistNode], bool]) -> LeftistNode:
        """Слияние двух левосторонних деревьев"""

        # Алгоритм:
        # Так как дерево левостороннее, самая правая ветвь имеет наименьшую длину.
        # Туда и поместим сливаемый корень (или исходный, смотря чей приоритет выше).
        # Нарушения левосторонности будем исправлять по ходу путём перестановки левого и правого поддеревьев.
        if first is None:
            return second
        elif second is None:
            return first

        if comp(second, first):
            # Приоритет второго корня выше первого, а так как правое поддерево опускается до самого правого
            # нижнего места, нужно совершить обмен.
            first, second = second, first
        # Рекурсивно опускаем правое поддерево в правое свободное место
        first.right = LeftistHeap.merge_nodes(first.right, second, comp)

        # Могло возникнуть нарушение левосторонности
        if first.left is None:
            first.left = first.right
            first.right = None
        elif first.right.dist > first.left.dist:
            first.left, first.right = first.right, first.left

        # Пересчитываем dist и обновляем родителей
        if first.right is not None:
            first.dist = first.right.dist + 1
            first.right.parent = first
        first.left.parent = first

        return first

    @staticmethod
    def merge_iterable(values: Iterable, comp: Callable[[LeftistNode, LeftistNode], bool]) -> Tuple[LeftistNode, int]:
        """Построение кучи из Iterable за O(N)

        :param values: список элементов
        :param comp: функция сравнение двух элементов, обёрнутых в структуру LeftistNode
        :return: вершина дерева (кучи), type LeftistNode

        """
        nodes = deque()

        # Создаем len(values) куч из 1 элемента и добавляем их в очередь
        for val in values:
            n = LeftistNode(val, 0, None, None, None)
            nodes.append(n)
        size = len(nodes)

        # Берем две первые кучи, объединеям их и возвращаем результат в конец,
        # пока не остается едиснвтенная левосторонняя куча.
        while len(nodes) > 1:
            a = nodes.popleft()
            b = nodes.popleft()
            nodes.append(LeftistHeap.merge_nodes(a, b, comp))
        return nodes.pop(), size

    def merge(self, *args: Iterable):
        """Объединение элементов из *args в кучу"""
        for arg in args:
            if not arg:
                continue
            if isinstance(arg, LeftistHeap):
                head, size = arg._head, arg.size
            else:
                head, size = self.merge_iterable(arg, self._compn)
            self.size += size

            self._head = self.merge_nodes(self._head, head, self._compn)

    def push(self, item: Any):
        """Добавление элемента в кучу"""
        # Время работы приходится на слияение с кучей из одного элемента и не превосходит O(log(N))
        node = LeftistNode(item, 0, None, None, None)

        self.size += 1

        self._head = LeftistHeap.merge_nodes(self._head, node, self._compn)

    def pop(self) -> Any:
        """Удалить и вернуть верхний элемент"""
        if self.size == 0:
            raise HeapIndexError("pop from empty heap")
        self.size -= 1

        # Ставим на место удаляемого верхнего элемента результат слияния двух его потомков
        # Время операции приходится на слияние и равно O(log(N))
        to_return = self._head
        self._head = LeftistHeap.merge_nodes(self._head.left, self._head.right, self._compn)
        return to_return.value

    def replace(self, item: Any) -> Any:
        """Удалить и вернуть верхний элемент, а потом вставить новый"""
        # Время работы зависит от операции просеивания вниз или слияния и составляет O(log(N))
        if self.size == 0:
            raise HeapIndexError("pop from empty heap")

        node = LeftistNode(item, 0, None, None, None)
        to_return = self._head.value

        if self.size == 1:
            # В куче только один элемент, поэтому просто заменяем его
            self._head = node
        elif self._head.right is None:
            # У корневого элемента нет правых потомков, поэтому поставим на его место
            # результат слияния левого потомка и нового элемента
            self._head = LeftistHeap.merge_nodes(self._head.left, node, self._compn)
        else:
            # У корневого элемента существуют оба потомка
            self._head.value = item      # Заменяем значение корневого элемента
            self._siftdown(self._head)   # Просеиваем новое значение вниз, восстанавливая свойство кучи

        return to_return

    def pushpop(self, item: Any) -> Any:
        """Эквивалент последовательному вызову push() и pop(), выполняющий меньше операций"""
        node = LeftistNode(item, 0, None, None, None)

        if self.size == 0:
            return item

        if self._compn(node, self._head):
            # Приоритет нового элемента выше текущего верхнего, поэтому просто вернем новичка
            return item
        elif self.size == 1:
            # В куче один элемент, поэтому просто заменяем его
            to_return = self._head.value
            self._head = node
            return to_return
        else:
            to_return = self._head.value
            self._head.value = item      # Заменяем значение корневого элемента
            self._siftdown(self._head)   # Просеиваем новое значение вниз, восстанавливая свойство кучи
            return to_return

    def find(self, val: Any) -> Union[None, LeftistNode]:
        """Поиск узла, хранящего значение val.

        Время работы составляет O(n), но алгоритм отсекает некоторые поддеревья,
        в которых заведомо нет искомого элемента.
        """
        if self._head is None:
            return

        stack = [self._head]
        while stack:
            node = stack.pop()
            if node.value == val:
                return node

            # Мы отправляемся в поддерево только в том случае, когда приоритет корня поддерева
            # больше приоритета искомого элемента, т.к. в противном случае искомого элемента там быть не может
            # по свойству кучи.
            if not self.comp(node.value, val):
                continue

            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)

    def update(self, old: Any, new: Any):
        """Заменить первый найденный элемент old элементом new"""
        if new == old:
            return

        node = self.find(old)
        if node is not None:
            node.value = new   # Заменяем значение в узле
            # Восстанавливаем свойство кучи
            self._siftup(node)
            self._siftdown(node)

    def remove(self, val: Any):
        """Удаление узла с заданным значением
        Алгоритм:
        1) Найдем узел со значением val и заменим его результатом слияния левого и правого поддеревьев.
        2) Пойдем вверх от предка вырезанной вершины, восстанавливая свойство левосторонности.
        Время работы: O(N) на поиск и O(log(N)) на удаление.

        """
        node = self.find(val)
        if node is None:
            return
        self.size -= 1

        # Потомки "забывают" родителя
        if node.left:
            node.left.parent = None
        if node.right:
            node.right.parent = None

        # Сливаем поддеревья удаляемого узла
        merged_subtrees = LeftistHeap.merge_nodes(node.left, node.right, self._compn)

        parent = node.parent
        if parent is None:
            # Удаляемый узел оказался корнем
            self._head = merged_subtrees
            return
        elif parent.left == node:
            parent.left = merged_subtrees
        else:
            parent.right = merged_subtrees

        if merged_subtrees is not None:
            merged_subtrees.parent = parent

        # Идем вверх, восстанавливая свойство левосторонности и обновляя dist
        node = parent
        while node is not None:
            left_dist = right_dist = -1
            if node.left is not None:
                left_dist = node.left.dist
            if node.right is not None:
                right_dist = node.right.dist

            new_dist = min(left_dist, right_dist) + 1   # Новое значение dist текущей вершины

            if left_dist < right_dist:
                node.left, node.right = node.right, node.left

            if node.dist == new_dist:
                break  # Удаление узла больше не влияет на dist - свойство левосторонности восстановлено

            node.dist = new_dist
            node = node.parent

    def _siftup(self, node: LeftistNode):
        """Просеивание вверх

        Перемещаются только значения в узлах, структура дерева остается прежней.
        Время работы O(log(N)).
        """
        while node.parent is not None:
            if self._compn(node.parent, node):
                break   # Свойство кучи соблюдено
            node.parent.value, node.value = node.value, node.parent.value
            node = node.parent

    def _siftdown(self, node: LeftistNode):
        """Просеивание вниз

        Перемещаются только значения в узлах, структура дерева остается прежней.
        Время работы O(log(N)).
        """
        # Двигаемся вниз от корня к потомкам, меня местами значение родителя со значением потомка,
        # если нарушено свойство кучи. Структура узлов не изменяется, свапаются только значения,
        # поэтому нарушение свойства левосторонности невозможно.
        while node.left is not None:
            # Находим самого приоритетного потомка
            child = node.left
            if node.right is not None and self._compn(node.right, child):
                child = node.right

            if self._compn(node, child):
                break   # Приоритет родителя выше -> свойство кучи соблюдено

            node.value, child.value = child.value, node.value   # Меняем местами значения в узлах
            node = child

    def __add__(self, other: Iterable) -> "LeftistHeap":
        new = deepcopy(self)
        new.merge(other)
        return new

    def __iadd__(self, other: Iterable) -> "LeftistHeap":
        self.merge(other)
        return self

    def __str__(self):
        return "LeftistHeap(size={}, head={})".format(self.size, self.head)

    def __bool__(self):
        return self.size > 0

    def __len__(self):
        return self.size

    def __contains__(self, item: Any):
        return self.find(item) is not None

    def __iter__(self):
        return self.items


class MinLeftistHeap(LeftistHeap):
    def __init__(self, values=None):
        super(MinLeftistHeap, self).__init__(values=values, comp=lt)

    def __str__(self):
        return "MinLeftistHeap(size={}, head={})".format(self.size, self.head)


class MaxLeftistHeap(LeftistHeap):
    def __init__(self, values=None):
        super(MaxLeftistHeap, self).__init__(values=values, comp=gt)

    def __str__(self):
        return "MaxLeftistHeap(size={}, head={})".format(self.size, self.head)


def main():
    heap = LeftistHeap()
    heap.merge([1, 2, 3], LeftistHeap([4, 5]))    # В кучу добавятся элементы 1-5
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
