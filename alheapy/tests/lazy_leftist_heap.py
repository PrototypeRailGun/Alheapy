# -*- coding: utf-8 -*-
"""Ленивая левосторонняя куча.
# Инициализация
heap = LazyLeftistHeap(values=None, comp=lt) # Создание кучи из элементов value с использованием функции сравнения comp.
По умолчанию comp=lt, поэтому верхним элементом будет минимальным. В общем случае функция comp принимает
два аргумента - элемента и возвращает True, если приоритет первого больше, чем у второго или False, если наоборот.
heap.merge([1, 2, 3], LazyLeftistHeap([4, 5]))   # В кучу добавятся элементы 1-5
heap.head                                        # Напечатает 1 - верхний элемент кучи
heap.push(-5)                                    # Добавление в кучу элемента -5
heap.head                                        # Напечатает -5
head = heap.pop()                                # Взятие верхнего элемента из кучи, head = -5
head = heap.replace(6)                           # Замена верхнего элемента значением 6, head = 1
head = pushpop(7)                                # Эквивалент последовательному вызову push() и pop(), head = 2
heap.remove(3)                                   # Удаление первого найденного элемента со значением 3
heap.items                                       # Итератор по элементам кучи, значения перебираются обходом в ширину
MinLazyLeftistHeap и MaxLazyLeftistHeap являются частными случаями LazyLeftistHeap с функциями сравнения
соотв. comp=lt и comp=gt.
# Важная деталь реализации:
Все значения, содержащиеся в куче, внутренне хранятся в виде узлов LazyLeftistNode.
@dataclass
class LazyLeftistNode:
    value: Any                               # Ценное значение (элемент)
    dist: int                                # Расстояние до ближайшего свободного места
    is_empty: bool                           # Является ли узел пустым
    parent: Union["LeftistNode", None]       # Родитель
    left: Union["LeftistNode", None]         # Левый потомок
    right: Union["LeftistNode", None]        # Правый потомок
# Про ленивую левостороннюю кучу
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
Ленивая левосторонняя куча отличается от неленивой реализацией методов.
# Поддерживаемые операции
----------------------------------------------------------------
 № | Базовые поддерживаемые операции           |   Сложность   |
---|-------------------------------------------|---------------|
 1 | Вставка                                   | O(1)          |
----------------------------------------------------------------
 2 | Слияние двух левосторонних куч            | O(1)          |
----------------------------------------------------------------
 3 | Удаление произвольного элемента           | O(1)          |
----------------------------------------------------------------
 4 | Построение lazy leftist heap              | O(N)          |
----------------------------------------------------------------
# Поиск верхнего элемента и вставка
Сложность выполнения операций НАЙТИ ВЕРХНИЙ ЭЛЕМЕНТ и ИЗВЛЕЧЬ ВЕРХНИЙ ЭЛЕМЕНТ является "расплатой за лень"
и составляет O(k * max(1, log(N)/(k+1)), где k - количество верхних пустых узлов.
Пустые узлы - узлы со значением is_empty=True, и в "шапке" кучи они появляются всвязи с ленивостью операций
слияния и вставки. Также пустые узлы могут возникать в "теле" кучи, т.к. операция удаление лишь помечает узел пустым.
(Ленивое) слияние: заводится новый пустой узел, детьми которого становятся корни сливаемых куч.
Вставка: происходит ленивое слияние исходной кучи и кучи из одного элемента.
Построение ленивой левосторонней кучи аналогично построению неленивой.
"""

__all__ = ["LazyLeftistHeap", "MinLazyLeftistHeap", "MaxLazyLeftistHeap"]


from dataclasses import dataclass
from copy import deepcopy
from operator import lt, gt
from typing import Any, Tuple, Union, Callable
from collections import deque
from collections.abc import Iterable

from alheapy._heap import Heap, HeapIndexError


@dataclass
class LazyLeftistNode:
    """Узел, хранящий один элемент ленивой левосторонней кучи.
    :attr value: данные, хранящиеся в узле
    :attr: dist - расстояние до ближайшего свободного места, 0 если узел имеет 0 или 1 потомков.
    :is_empty: является ли узел пустым
    :attr parent: родительская вершина
    :attr left: левый потомок
    :attr right: правый потомок
    """
    value: Any
    dist: int
    is_empty: bool
    parent: Union["LazyLeftistNode", None]
    left: Union["LazyLeftistNode", None]
    right: Union["LazyLeftistNode", None]


class LazyLeftistHeap(Heap):
    """Ленивая левосторонняя куча.
    # Все элементы кучи внутри хранятся в виде узлов LazyLeftistNode
    # Публичные атрибуты и @property:
    heap.head - верхний (самый приоритетный элемент)
    heap.items - итератор по списку элементов кучи
    # Публичные методы:
    heap.push(item) - добавление нового элемента
    heap.pop(idx) - извлечь верхний элемент
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
    LazyLeftistHeap.merge_iterable(*args) - построение кучи из Iterable за O(N)
    LazyLeftistHeap.lazy_merge(first, second) - ленивое слияние двух узлов.
    """
    def __init__(self, values: Iterable = None, comp: Callable[[Any, Any], bool] = lt):
        self._compfunc = comp
        self._head: Union[LazyLeftistNode, None] = None   # Верхний узел
        self.size = 0   # Кол-во непустых узлов в куче
        if values:
            self.merge(values)

    def _top_node(self) -> Union[None, LazyLeftistNode]:
        """Поиск узла, содержащего самый приоритетный элемент"""
        if self._head is None:
            return

        top_priority = None

        # Обходом в глубину найдем все верхние непустые узлы, один из них имеет наибольший приоритет
        queue = deque([self._head])
        while queue:
            node = queue.popleft()
            if not node.is_empty:
                if top_priority is None:
                    top_priority = node
                elif self._compn(node, top_priority):
                    top_priority = node
                continue

            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)

        return top_priority

    @property
    def head(self) -> Any:
        """Верхний элемент"""
        top = self._top_node()
        if top is not None:
            return top.value

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

            if not node.is_empty:
                yield node.value

    def comp(self, first: Any, second: Any) -> bool:
        """True если приоритет first больше (или равен) приоритета second, и False иначе"""
        return self._compfunc(first, second)

    def _compn(self, first: LazyLeftistNode, second: LazyLeftistNode) -> bool:
        """True если приоритет first больше (или равен) приоритета second, и False иначе"""
        return self._compfunc(first.value, second.value)

    @staticmethod
    def merge_nodes(first: LazyLeftistNode, second: LazyLeftistNode,
                    comp: Callable[[LazyLeftistNode, LazyLeftistNode], bool]) -> LazyLeftistNode:
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
        first.right = LazyLeftistHeap.merge_nodes(first.right, second, comp)

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
    def merge_iterable(
            values: Iterable, comp: Callable[[LazyLeftistNode, LazyLeftistNode], bool]) -> Tuple[LazyLeftistNode, int]:
        """Построение кучи из Iterable за O(N)
        :param values: список элементов
        :param comp: функция сравнение двух элементов, обёрнутых в структуру LazyLeftistNode
        :return: вершина дерева (кучи), type LazyLeftistNode
        """
        nodes = deque()

        # Создаем len(values) куч из 1 элемента и добавляем их в очередь
        for val in values:
            n = LazyLeftistNode(val, 0, False, None, None, None)
            nodes.append(n)
        size = len(nodes)

        # Берем две первые кучи, объединеям их и возвращаем результат в конец,
        # пока не остается едиснвтенная левосторонняя куча.
        while len(nodes) > 1:
            a = nodes.popleft()
            b = nodes.popleft()
            nodes.append(LazyLeftistHeap.merge_nodes(a, b, comp))
        return nodes.pop(), size

    @staticmethod
    def lazy_merge(first: LazyLeftistNode, second: LazyLeftistNode) -> LazyLeftistNode:
        """Ленивое объединение в кучу
        Заводим пустой корневой узел, сыновьями которого становятся корневые элементы
        сливаемых куч. Время работы такой ленивой операции O(1).
        """
        if first.dist >= second.dist:
            head = LazyLeftistNode(None, second.dist+1, True, None, first, second)
        else:
            head = LazyLeftistNode(None, first.dist+1, True, None, second, first)
        first.parent, second.parent = head, head
        return head

    def merge(self, *args: Iterable):
        """Слияние кучи с элементами из *args"""
        for arg in args:
            if not arg:
                continue
            if isinstance(arg, LazyLeftistHeap):
                head, size = arg._head, arg.size
            else:
                head, size = self.merge_iterable(arg, self._compn)
                if self._head is None:
                    # Создание кучи
                    self.size = size
                    self._head = head
                    return

            self.size += size
            self._head = self.lazy_merge(self._head, head)

    def push(self, item: Any):
        """Добавление элемента в кучу"""
        node = LazyLeftistNode(item, 0, False, None, None, None)
        self.size += 1
        if self._head is None:
            self._head = node
        else:
            self._head = LazyLeftistHeap.lazy_merge(self._head, node)

    def pop(self) -> Any:
        """Удалить и вернуть верхний элемент
        Время работы приходится на поиск текущего верхнего элемента и равно
         O(k * max(1, log(N)/(k+1)), где k - количество верхних пустых узлов.
        """
        if self.size == 0:
            raise HeapIndexError("pop from empty heap")
        self.size -= 1
        top = self._top_node()
        top.is_empty = True
        return top.value

    def replace(self, item: Any) -> Any:
        """Удалить и вернуть верхний элемент, а потом вставить новый.
        Время работы приходится на поиск текущего верхнего элемента и равно
         O(k * max(1, log(N)/(k+1)), где k - количество верхних пустых узлов.
        """
        if self.size == 0:
            raise HeapIndexError("pop from empty heap")

        to_return = self.pop()
        self.push(item)
        return to_return

    def pushpop(self, item: Any) -> Any:
        """Эквивалент последовательному вызову push() и pop(), выполняющий меньше операций
        Время работы приходится на поиск текущего верхнего элемента и равно
         O(k * max(1, log(N)/(k+1)), где k - количество верхних пустых узлов.
        """
        top = self._top_node()
        if top is None:
            return item
        if self.comp(item, top.value):
            return item

        node = LazyLeftistNode(item, 0, False, None, None, None)
        top.is_empty = True
        self._head = LazyLeftistHeap.lazy_merge(self._head, node)
        return top.value

    def find(self, val: Any) -> Union[None, LazyLeftistNode]:
        """Поиск узла, хранящего значение val.
        Время работы составляет O(N), но алгоритм отсекает некоторые поддеревья,
        в которых заведомо нет искомого элемента.
        """
        if self._head is None:
            return

        stack = [self._head]
        while stack:
            node = stack.pop()
            if not node.is_empty and node.value == val:
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
        """Заменить первый найденный элемент old элементом new
        Время работы приходится на поиск и составляет O(N), а потом еще O(log(N)) работы на восстанволение
        свойств кучи.
        """
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
        Если узел со значением val существует, мы находим его, а затем помечаем пустым.
        Время работы приходится на поиск и составляет O(N), но само удаление требует лишь O(1) операций.
        """
        node = self.find(val)
        if node is not None:
            self.size -= 1
            node.is_empty = True

    def _siftup(self, node: LazyLeftistNode):
        """Просеивание вверх
        Перемещаются только значения в узлах, структура дерева остается прежней.
        Время работы O(log(N)).
        """
        while node.parent is not None:
            # Если родительский узел помечен как пустой, то мы его просто игнорируем и идем дальше вверх
            if not node.parent.is_empty:
                if self._compn(node.parent, node):
                    break   # Свойство кучи соблюдено
                node.parent.value, node.value = node.value, node.parent.value
            node = node.parent

    def _siftdown(self, node: LazyLeftistNode):
        """Просеивание вниз
        Перемещаются только значения в узлах, структура дерева остается прежней.
        Но вместе с перемещением значений нужно обновлять метки is_empty.
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

            # Меняем местами значения в узлах и соответсвующим образом обновляем метки
            # Ведь если мы перемещаем "несуществующее" значение в другой узел, в том узле is_empty
            # теперь должно быть True, и наоборот, если в "пустой" узел переносится существующее значение,
            # то его метка is_empty должна быть равна False.
            node.value, child.value = child.value, node.value
            node.is_empty, child.is_empty = child.is_empty, node.is_empty
            node = child

    def __add__(self, other: Iterable) -> "LazyLeftistHeap":
        new = deepcopy(self)
        new.merge(other)
        return new

    def __iadd__(self, other: Iterable) -> "LazyLeftistHeap":
        self.merge(other)
        return self

    def __str__(self):
        return "LazyLeftistHeap(size={}, head={})".format(self.size, self.head)

    def __bool__(self):
        return self.size > 0

    def __len__(self):
        return self.size

    def __contains__(self, item: Any):
        return self.find(item) is not None

    def __iter__(self):
        return self.items


class MinLazyLeftistHeap(LazyLeftistHeap):
    def __init__(self, values=None):
        super(MinLazyLeftistHeap, self).__init__(values=values, comp=lt)

    def __str__(self):
        return "MinLazyLeftistHeap(size={}, head={})".format(self.size, self.head)


class MaxLazyLeftistHeap(LazyLeftistHeap):
    def __init__(self, values=None):
        super(MaxLazyLeftistHeap, self).__init__(values=values, comp=gt)

    def __str__(self):
        return "MaxLazyLeftistHeap(size={}, head={})".format(self.size, self.head)


def main():
    heap = LazyLeftistHeap()
    heap.merge([1, 2, 3], LazyLeftistHeap([4, 5]))   # В кучу добавятся элементы 1-5
    print(heap.head)                                 # Напечатает 1 - верхний элемент кучи
    heap.push(-5)                                    # Добавление в кучу элемента -5
    print(heap.head)                                 # Напечатает -5
    top = heap.pop()                                 # Взятие верхнего элемента из кучи, head = -5
    print(top)
    top = heap.replace(6)                            # Замена верхнего элемента значением 6, head = 1
    print(top)
    top = heap.pushpop(7)                            # Эквивалент последовательному вызову push() и pop(), head = 2
    print(top)


if __name__ == "__main__":
    main()
