# -*- coding: utf-8 -*-
"""Бинарная куча

Инициализация и параметры:
BinaryHeap(values=v, comp=compfunc)
MinBinaryHeap(values=v, comp=compfunc)
MaxBinaryHeap(values=v, comp=compfunc)

# Нумерация элементов начинается с 0
# Для элемента p потомками являются 2*p + 1, 2*p + 2
# Для элемента k родителем является (k - 1) // 2

Использование бинарной кучи на примере MinBinaryHeap.
MaxBinaryHeap и BinaryHeap обладают тем же набором операций.

heap = MinBinaryHeap()   # Пустая минимальная бинарная куча
heap.push(1)             # Добавление элемента 1 в кучу
heap.push(2)             # Добавление элемента 2 в кучу
heap.pop()               # Удаление наименьшего элемента
print(heap)              # MinBinaryHeap([2])
heap.replace(3)          # Удаляет верхний элемент и добавляет новый
print(heap)              # MinBinaryHeap([3])
heap.pushpop(4)          # MinBinaryHeap([4]) - эквивалентно последовательному вызову heap.push(4) и heap.pop(),
                         # но работает быстрее.

print(heap + MinBinaryHeap([1, 2, 3, 4, 5]))  # [1, 3, 2, 4, 4, 5] - список, хранящий элементы в виде кучи.
                                              # Того же результата можно достичь, просто прибавив к heap [1, 2, 3, 4, 5]

# Дополнение кучи новыми элементами (можно прибавлять другую кучу, множество и т.д.)
heap += [1, 2, 3, 4, 5]
print(heap)   # MinBinaryHeap([1, 3, 2, 4, 4, 5])

# Слияние кучи со всеми наборами элементов из *args за линейное время
heap.dmerge([0], {-1}, (-2, -3), MaxBinaryHeap([-4, -5]))
print(heap)   # MinBinaryHeap([-5, -3, -4, 0, -2, 1, 2, 4, 3, 4, -1, 5])

# Вывести верхний элемент  кучи можно двумя способами
print(heap[0], heap.head)  ->  -5, -5
heap.items - итератор по элементам кучи
heap.children(1)   # [0, -2] - потомки элемента с номером 1

Перебор всех элементов кучи в том порядке, в котором они хранятся в self._items
for i in heap:
    print(i)

# !!! Если элементы не имеют определенного поведения для операторов сравнения, то стоит использовать
BinaryHeap с указанием особой функции сравнения при инициализации. BinaryHeap(values, _compn=special_func)

########################################################################################################################

Немного теории:

Двоичная куча представляет собой полное бинарное (коэффициентом ветвления 2) дерево,
для которого выполняется основное свойство кучи (инвариант):приоритет каждой вершины не меньше приоритетов её потомков.
В простейшем случае приоритет каждой вершины можно считать равным её значению.
В таком случае структура называется max-heap, поскольку корень поддерева является максимумом из
значений элементов поддерева. Если же самый большой приоритет отдавать минимальным элементам,
то такая куча будет называться min-hea. Дерево называется полным бинарным, если у каждой вершины есть
не более двух потомков, а заполнение уровней вершин идет сверху вниз (в пределах одного уровня – слева направо).

Вот пример минимальной бинарной кучи (min-heap), хранящей числа [0, 30]

                                  0

                  1                                 2

          3               4                5               6

      7       8       9       10      11      12      13      14

    15 16   17 18   19 20   21 22   23 24   25 26   27 28   29 30

В данной реализации элементы кучи хранятся в списке (list), где для i-го элемента потомками являются элементы
2*i + 1 и 2*i + 2, а родителем элемент с индексом (i - 1) // 2

----------------------------------------------------------------
 № | Базовые поддерживаемые операции           |   Сложность   |
---|-------------------------------------------|---------------|
 1 | Получение верхнего                        | O(1)          |
   |(самого приоритетного элемента)            |               |
----------------------------------------------------------------
 2 | Вставка                                   | O(log(N))     |
----------------------------------------------------------------
 3 | Удаление верхнего элемента                | O(log(N))     |
----------------------------------------------------------------
 4 | Построение binary heap (heapify)          | O(N)          |
----------------------------------------------------------------

Остальные операции, доступные в классе BinaryHeap, опираются на приведенные выше.
Список всех методов и атрибутов в шапке класса.

########################################################################################################################

Больше деталей реализации:

Класс BinaryHeap представляет произвольную бинарную кучу, при инициализации дополнительно принимает параметр
comp(first, second) - функцию, которая проверяет, больше ли приоритет первого аргумента чем у второго.
MinBinaryHeap и MaxBinaryHeap наследуются от BinaryHeap и передают в родительский __init__ соответственно
comp=lt и comp=gt.
BinaryHeap наследуется от DHeap (кучи с произвольным коэффициентом ветвления), которая, в свою очередь, наследуется от
абстрактной кучи Heap, которая определяет абстрактные @property .head и .items - соответсвенно верхний элемент и
список всех элементов кучи (в других кучах, например, биномиальной, items имеет более сложную структуру),
и абстрактный метод _compn, который нужен для сравнения приоритетов двух элементов, как было сказано выше.
Эти три объявления на общем уровне нужны для поддержаний общего интерфейса и стиля работы для всех куч.
Также в классе Heap определены 6 операций сравнения, которое происходит по приоритетам верхних элементов,
!!! с использованием функции сравнения левого операнда !!!
Также можно сравнивать с произвольной коллекцией, тогда верхним элементом будет считаться нулевой.

Для более глубокого понимания архитектуры стоит ознакомиться с кодом.

"""

__all__ = ["BinaryHeap", "MaxBinaryHeap", "MinBinaryHeap", "binary_merge"]


from operator import lt, gt
from collections.abc import Iterable
from typing import Any, Callable

from alheapy.dheap import DHeap, dmerge


class BinaryHeap(DHeap):
    """Binary Heap

    Основные сведения:
    1) BinaryHeap(values) является оберткой для входных данных, поэтому любые изменения в куче
        затрагивают исходный список.
    2) При инициализации можно указать параметр _compn (по умолчанию =operator.lt), этот
        эта функция нужна для сравнения двух элементов: _compn(first, second)
        (True если приоритет first больше (или равен), чем у second и False в противном случае)


    Публичные атрибуты и @property:
    heap.head - верхний (самый приоритетный элемент)
    heap.tail - элемент с наименьшим приоритетом
    heap.items - итератор по списку элементов кучи

    Публичные методы:
    heap.push(item) - добавление нового элемента
    heap.pop(idx=0) - удалить и вернуть элемент в позиции idx, по умолчанию будет удален верхний элемент
    heap.replace(item) - удалить и вернуть верхний элемент, а потом вставить новый
    heap.pushpop(item) - эквивалентно последовательному вызову heap.push() и heap.pop(),
        но использует лишь одну операцию просевивания, что повышает эффективность почти в 2 раза.
    heap.find(val) - с учетом val, найти номер элемента с таким значением  или вернуть -1
        в случае отсутсвия
    heap.update(idx, item) - заменить элемент в позиции idx элементом item
    heap.update(old, new) - заменить первое вхождение элемента new элементом old
    heap.remove(value) - удалить первый найденный элемент с номером
    heap.merge(*args) - добавление в кучу элементов из *args
    heap._compn(a, b) - True если приоритет a больше (или равен), чем у b и False в противном случае

    Протоколы контейнера:
    heap[i]         # Элемент под номером i в списке элементов кучи
    heap[i] = val   # Заменить элемент с индексом i элементом item с сохранением инварианта
    del heap[i]     # Удалить элемент с индексом i с сохранением инварианта

    """
    def __init__(self, values: Iterable = None, comp: Callable[[Any, Any], bool] = lt):
        super(BinaryHeap, self).__init__(2, values=values, comp=comp)

    def __str__(self):
        return "BinaryHeap({})".format(self._items)


class MinBinaryHeap(BinaryHeap):
    """Минимальная бинарная куча"""
    def __init__(self, values: Iterable = None):
        super(MinBinaryHeap, self).__init__(values=values, comp=lt)

    def __str__(self):
        return "MinBinaryHeap({})".format(self._items)


class MaxBinaryHeap(BinaryHeap):
    """Максимальная бинарная куча"""
    def __init__(self, values: Iterable = None):
        super(MaxBinaryHeap, self).__init__(values=values, comp=gt)

    def __str__(self):
        return "MaxBinaryHeap({})".format(self._items)


def binary_merge(*args: Iterable, comp=lt):
    """Слияние нескольких бинарных куч или других коллекций в новую бинарную кучу.
    Результатом функции будет список с бинарной кучей.
    Время работы: O(n), где n - количество элементов в новой куче.
    """
    return dmerge(*args, d=2, comp=comp)


def main():
    heap = MinBinaryHeap()  # Пустая минимальная бинарная куча
    heap.push(1)            # Добавление элемента 1 в кучу
    heap.push(2)            # Добавление элемента 2 в кучу
    heap.pop()              # Удаление наименьшего элемента
    print(heap)             # MinBinaryHeap([2])
    heap.replace(3)         # Удаляет верхний элемент и добавляет новый
    print(heap)             # MinBinaryHeap([3])
    heap.pushpop(4)         # эквивалентно последовательному вызову heap.push(4) и heap.pop(), но работает быстрее

    print(heap + MinBinaryHeap([1, 2, 3, 4, 5]))  # [1, 3, 2, 4, 4, 5] - список, хранящий элементы в виде кучи.
    # Того же результата можно достичь, просто прибавив к heap [1, 2, 3, 4, 5]

    # Дополнение кучи новыми элементами (можно прибавлять другую кучу, множество и т.д.)
    heap += [1, 2, 3, 4, 5]
    print(heap)  # MinBinaryHeap([1, 3, 2, 4, 4, 5])

    # Слияние кучи со всеми наборами элементов из *args за линейное время
    heap.merge([0], {-1}, (-2, -3), MaxBinaryHeap([-4, -5]))
    print(heap)  # MinBinaryHeap([-5, -3, -4, 0, -2, 1, 2, 4, 3, 4, -1, 5])

    # Вывести верхний элемент  кучи можно двумя способами
    print(heap[0], heap.head)   # ->  -5, -5
    print(heap)
    print(heap.children(1))


if __name__ == "__main__":
    main()
