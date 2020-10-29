# -*- coding: utf-8 -*-
"""d-куча

# Инициализация
DHeap(d, values=v, comp=compfunc)
MinDHeap(values=v, comp=compfunc)
MaxDHeap(values=v, comp=compfunc)

# Нумерация элементов начинается с 0
# Для элемента p потомками являются d*p + 1, d*p + 2, ... , d*p + d, где d - число потомков у элемента p
# Для элемента k родителем является (k - 1) // d

Использование d-кучи на примере MinDHeap.
MaxDHeap и DHeap обладают тем же набором операций.
# Все операции происходят на месте, поэтому передаваемый при инициализации список скорее всего окажется изменен

heap = MinDHeap(4)       # Пустая минимальная d-куча с коэффициентом ветвления 4
heap.push(1)             # Добавление элемента 1 в кучу
heap.push(2)             # Добавление элемента 2 в кучу
heap.pop()               # Удаление наименьшего элемента
print(heap)              # MinDHeap([2], d=4)
heap.replace(3)          # Удаляет верхний элемент и добавляет новый
print(heap)              # MinDHeap([3], d=4)
heap.pushpop(4)          # MinDHeap([4], d=4) - эквивалентно последовательному вызову heap.push(4) и heap.pop(),
                         # но работает быстрее.

print(heap + MinDHeap(3, [1, 2, 3, 4, 5]))  # [1, 3, 2, 4, 4, 5] - список, хранящий элементы в виде кучи.
                                            # Того же результата можно достичь, просто прибавив к heap [1, 2, 3, 4, 5]

# Дополнение кучи новыми элементами (можно прибавлять другую кучу, множество и т.д.)
heap += [1, 2, 3, 4, 5]
print(heap)   # MinDHeap([1, 4, 2, 3, 4, 5], d=4)

# Слияние кучи со всеми наборами элементов из *args за линейное время
heap.dmerge([0], {-1}, (-2, -3), MaxDHeap(1, [-4, -5]))
print(heap)   # MinDHeap([-5, -1, -4, 3, 4, 5, 4, 1, 0, 2, -2, -3], d=4)

# Вывести верхний элемент кучи можно двумя способами
print(heap[0], heap.head)  ->  -5, -5
heap.items   # Итератор по элементам кучи
heap.children(1)   # [5, 4, 1, 0] - потомки элемента с номером 1

Перебор всех элементов кучи в том порядке, в котором они хранятся в self._items
for i in heap:
    print(i)

Расширенные операции кучи:
heap.pop(1)              # Удаление элемента с индексом 1
heap.remove(5)           # Удаление из кучи первого найденного элемента со значением 5
heap[3] = 333            # Заменить значение третьего элемента на 333
del heap[1]              # Удалить элемент с индексом 1 из кучи

heap.find(-5)            # Поиск элемента со значением -5

# !!! Если элементы не имеют определенного поведения для операторов сравнения, то стоит использовать
DHeap с указанием особой функции сравнения при инициализации. DHeap(d, values, _compn=special_func)

С указанием d=2 получится бинарная куча, с d=1 - отсортированный массив.

########################################################################################################################

Немного теории:

d-куча представляет собой полное d-дерево с коэффициентом ветвления d,
для которого выполняется основное свойство кучи (инвариант): приоритет каждой вершины не меньше приоритетов её потомков.
В простейшем случае приоритет каждой вершины можно считать равным её значению.
В таком случае структура называется max-hea, поскольку корень поддерева является максимумом из
значений элементов поддерева. Если же самый большой приоритет отдавать минимальным элементам,
то такая куча будет называться min-heap. d-дерево называется полным, если у каждой вершины есть не более d потомков,
а заполнение уровней вершин идет сверху вниз (в пределах одного уровня – слева направо).

Вот пример минимальной d-кучи (min-heap) с коэффициентом ветвления 4, хранящей числа [0, 20]

                                     0
          1                2                  3                 4
     5  6   7  8     9  10  11  12      13  14  15  16    17  18  19 20

В данной реализации элементы кучи хранятся в списке (list), где для i-го элемента потомками являются элементы
d*i + 1, d*i + 2, ..., d*i + d,  а родителем элемент с индексом (i - 1) // d

----------------------------------------------------------------
 № | Базовые поддерживаемые операции           |   Сложность   |
---|-------------------------------------------|---------------|
 1 | Получение верхнего                        | O(1)          |
   |(самого приоритетного) элемента            |               |
----------------------------------------------------------------
 2 | Вставка                                   | O(log(d, N))  |
----------------------------------------------------------------
 3 | Извлечение верхнего элемента              | O(d*log(d, N) |
----------------------------------------------------------------
 4 | Построение d-heap                         | O(N)          |
----------------------------------------------------------------

Остальные операции, реализованные в классе DHeap, опираются на приведенные выше.

Преимущество d-кучи над бинарной в том, что
d-куча выполняет просеивание вверх в log(d, n) / log(2, n) раз быстрее, что также ускоряет push() и объединение в кучу.

Но стоит понимать, что с увеличением d растут затраты на каждом шаге просеивания вниз. Если для бинарной кучи мы
сравниваем всего-лишь двух потомков, то теперь мы проходим все d потомков. Это увеличивает время работы просеивания вниз
и, следовательно, pop() в log(d, n) / log(2, n) * d/2 раз.

########################################################################################################################

Больше деталей реализации:

Класс DHeap представляет произвольную d-кучу, при инициализации дополнительно принимает параметр
comp(first, second) - функцию, которая проверяет, больше ли приоритет первого аргумента чем у второго.
MinDHeap и MaxDHeap наследуются от DHeap и передают в родительский __init__ соответственно comp=lt и comp=gt.
DHeap наследуется от абстрактной кучи Heap, которая определяет абстрактные @property .head и .items - соответсвенно
верхний элемент и список всех элементов кучи (в других кучах, например, биномиальной,
items имеет более сложную структуру), и абстрактный метод comp, который нужен для сравнения приоритетов двух элементов,
как было сказано выше. Эти три объявления на общем уровне нужны для поддержаний общего интерфейса и стиля работы для
всех куч. Также в классе Heap определены 6 операций сравнения, которое происходит по приоритетам верхних элементов,
!!! с использованием функции сравнения левого операнда !!!
Также можно сравнивать с произвольной коллекцией, тогда верхним элементом будет считаться нулевой.
"""

__all__ = ["DHeap", "MinDHeap", "MaxDHeap", "dmerge"]


from operator import lt, gt
from copy import copy, deepcopy
from typing import Any, Callable
from collections.abc import Iterable

from alheapy._heap import Heap, HeapIndexError


class DHeap(Heap):
    """D-heap

    # Основные сведения:
    1) DHeap(d, values) является оберткой для входных данных, поэтому любые изменения в куче
        затрагивают исходный список.
    2) При инициализации можно указать параметр _compn (по умолчанию =operator.lt), этот
        эта функция нужна для сравнения двух элементов: _compn(first, second)
        (True если приоритет first больше (или равен), чем у second и False в противном случае)


    # Публичные атрибуты и @property:
    heap.head - верхний (самый приоритетный элемент)
    heap.tail - элемент с наименьшим приоритетом
    heap.items - итератор по списку элементов кучи

    # Публичные методы:
    heap.push(item) - добавление нового элемента
    heap.pop(idx=0) - удалить и вернуть элемент в позиции idx, по умолчанию будет удален верхний элемент
    heap.replace(item, idx=0) - заменить элемент в позиции idx элементом item и вернуть старый элемент
    heap.pushpop(item) - эквивалентно последовательному вызову heap.push() и heap.pop(),
        но использует лишь одну операцию просевивания, что повышает эффективность почти в 2 раза
    heap.find(val) - с учетом val, найти номер элемента с таким значением  или вернуть -1
        в случае отсутсвия
    heap.update(old, new) - заменить первое вхождение элемента new значением old
    heap.remove(val) - удалить первый найденный элемент со значением val
    heap.merge(*args) - добавление в кучу элементов из *args
    heap.comp(a, b) - True если приоритет a больше (или равен), чем у b и False в противном случае

    Протоколы контейнера:
    heap[i]         # Элемент под номером i в списке элементов кучи
    heap[i] = val   # Заменить элемент с индексом i элементом item с сохранением инварианта
    del heap[i]     # Удалить элемент с индексом i с сохранением инварианта

    """
    def __init__(self, d: int, values: Iterable = None, comp: Callable[[Any, Any], bool] = lt):
        """
        :param d: коэффициент ветвления (максимальное кол-во потомков у каждого элемента)
        :param values: элементы для объединения в кучу
        :param comp: функция сравнения двух элементов, если _compn=le, то куча будет минимальной,
            если _compn=ge, то максимальной. Также можно передавать пользовательскую функцию сравнения.

        """
        if d < 1:
            raise ValueError("the number of children cannot be less than 1")
        self.d = d

        self._items = values
        if self._items is None:
            self._items = []
        else:
            if not isinstance(self._items, list):
                self._items = list(self._items)
            self._items = _heapify(self._items, d=d, comp=comp)
        self._compfunc = comp

    @property
    def head(self) -> Any:
        """Верхний элемент"""
        return self._items[0] if self._items else None

    @property
    def tail(self) -> Any:
        """Элемент с наименьшим приоритетом"""
        # Так как d-куча не имеет "волшебного" метода нахождения минимального по приоритету элемента,
        # приходится перебирать все значения за линейное время
        if self._items is None:
            return None

        tail = self._items[0]
        for i in range(len(self._items)):
            if self.comp(tail, self._items[i]):
                tail = self._items[i]
        return tail

    @property
    def items(self) -> Iterable:
        """Элементы кучи"""
        return iter(self._items)

    def comp(self, first: Any, second: Any) -> bool:
        """True если приоритет first больше (или равен) приоритета second, и False иначе"""
        return self._compfunc(first, second)

    def _repair(self, idx: int):
        """Перемещаем элемент в позиции idx на правильное место"""
        if idx == 0:
            _siftdown(self._items, self.d, 0, comp=self.comp)

        if self.comp(self._items[idx], self._items[(idx - 1) // self.d]):
            # Восстанавливаем свойство кучи за O(log(d, N))
            _siftup(self._items, self.d, 0, idx, comp=self.comp)
        else:
            # Восстанавливаем свойство кучи за d * O(log(d, N))
            _siftdown(self._items, self.d, idx, comp=self.comp)

    def push(self, item: Any):
        """Добавление элемента item в кучу с сохранением инварианта.

        Время работы: O(log(d, N)), где N - число элементов в куче.
        """

        # Добавляем элемент в конец списка, сохраняя свойство полноты дерева,
        # а затем поднимаем его на законное место, восстанавливая инвариант
        self._items.append(item)
        _siftup(self._items, self.d, 0, len(self._items)-1, comp=self.comp)

    def pop(self, idx=0) -> Any:
        """Удаление элемента с номером idx из кучи, по умолчанию idx=0 - верхний элемент.

        Время работы: O(d*log(d, N)), где N - число элементов в куче.

        """
        if not isinstance(idx, int):
            raise TypeError("heap list index must be integers, not {}".format(type(idx)))

        if idx < 0:
            idx = len(self._items) + idx
        if len(self._items) <= idx or idx < 0:
            raise HeapIndexError("heap index out of range")

        last = self._items.pop()

        if idx != len(self._items) and self._items:
            to_return = self._items[idx]
            # Ставим последний элемент на место удаляемого и восстанавливаем инвариант кучи
            self._items[idx] = last
            self._repair(idx)
            return to_return
        return last

    def replace(self, item: Any, idx=0) -> Any:
        """Заменить элемент в позиции idx элементом item с сохранением инварианта"""
        if not isinstance(idx, int):
            raise TypeError("heap list index must be integers, not {}".format(type(idx)))

        if idx < 0:
            idx = len(self._items) + idx
        if len(self._items) <= idx or idx < 0:
            raise HeapIndexError("heap index out of range")

        to_return = self._items[idx]
        self._items[idx] = item
        self._repair(idx)
        return to_return

    def pushpop(self, item: Any) -> Any:
        """Более быстрый эквивалент последовательному вызову push и pop

        Если приоритет нового элемента item выше приоритета текущего верхнего,
        то нужно просто вернуть item, в противном случае pushpop выплняет эквивалентные
        методу replace шаги.

        """
        if self._items and self.comp(self._items[0], item):
            item, self._items[0] = self._items[0], item
            _siftdown(self._items, self.d, 0, comp=self.comp)
        return item

    def update(self, old: Any, new: Any):
        """Заменить первый найденный элемент old элементом new"""
        # O(N) на поиск и в худшем случае O(d*log(d, N)) на саму замену
        if new == old:
            return
        idx = self.find(old)
        if idx != -1:
            self.replace(new, idx)

    def find(self, val: Any) -> int:
        """Поиск элемента

        С учетом значения val, найти в списке кучи индекс элемента  с таким значением или вернуть -1,
        если такого элемента нет. Время работы не превышает O(N).
        """
        try:
            return self._items.index(val)
        except ValueError:
            return -1

    def remove(self, val: Any, idx=None):
        """Удаление первого найденного элемента со значением val

        Операция удаления в худшем случае требует O(N) времени на поиск и O(log(d, N)) или d*log(d, N)
        на восстановление инварианта.
        """
        if idx is None:
            idx = self.find(val)
        if idx == -1:
            return

        self.pop(idx)

    def merge(self, *args):
        """Слияние кучи со всеми наборами элементов из *args за линейное время"""
        self._items = dmerge(self._items, *args, d=self.d, comp=self.comp)

    def children(self, idx: int) -> list[Any]:
        """Потомки элемента под номером idx"""
        return self._items[self.d*idx+1:self.d*(idx+1)+1]

    def __add__(self, other):
        new = DHeap(self.d)
        new._items = dmerge(self, other, d=self.d, comp=self.comp)
        return new

    def __iadd__(self, other):
        self._items = dmerge(self, other, d=self.d, comp=self.comp)
        return self

    def __str__(self):
        return "DHeap({}, d={})".format(self._items, self.d)

    def __repr__(self):
        return str(self.items)

    def __bool__(self):
        return len(self._items) > 0

    def __len__(self):
        return len(self._items)

    def __contains__(self, item: Any):
        return item in self._items

    def __getitem__(self, idx: int):
        """Элемент под номером idx в списке элементов кучи"""
        if idx < 0:
            idx = len(self._items) + idx

        if not isinstance(idx, int):
            raise TypeError("heap list index must be integers, not {}".format(type(idx)))
        elif len(self._items) <= idx < 0:
            raise HeapIndexError("heap index out of range")
        return self._items[idx]

    def __setitem__(self, idx: int, item: Any):
        """Заменить элемент в позиции idx элементом item с сохранением инварианта"""
        self.replace(item, idx)

    def __delitem__(self, idx: int):
        """Удалить элемент в позиции idx с сохранением инварианта"""
        self.pop(idx)

    def __iter__(self):
        return iter(self._items)

    def __copy__(self):
        return DHeap(self.d, copy(self._items), comp=self._compfunc)

    def __deepcopy__(self, memo=None):
        return DHeap(self.d, deepcopy(self._items), comp=self._compfunc)


class MinDHeap(DHeap):
    """Минимальная d-куча."""
    def __init__(self, d: int, values: Iterable = None):
        super(MinDHeap, self).__init__(d, values=values, comp=lt)

    def __str__(self):
        return "MinDHeap({}, d={})".format(self._items, self.d)


class MaxDHeap(DHeap):
    """Максимальная d-куча."""
    def __init__(self, d: int, values: Iterable = None):
        super(MaxDHeap, self).__init__(d, values=values, comp=gt)

    def __str__(self):
        return "MaxDHeap({}, d={})".format(self._items, self.d)


def dmerge(*args: Iterable, d=2, comp=lt):
    """Слияние нескольких бинарных куч или других коллекций в новую d-кучу

    Результатом функции будет список с d-кучей.
    Время работы: O(n), где n - количество элементов в новой куче.

    """
    values = []

    for arg in args:
        if isinstance(arg, list):
            values.extend(arg)
        elif isinstance(arg, Heap):
            values.extend(arg.items)
        else:
            values.extend(list(arg))

    return _heapify(values, d, comp=comp)


def _siftup(heap: list[Any], d: int, rootpos: int, pos: int, comp=lt):
    """Просеять вверх за время O(log(d, N))

    :param heap: вся куча в виде списка элементов
    :param rootpos: индекс верхнего элемента в поддереве
    :param pos: индекс нового элемента
    :param comp: функция сравнения двух элементов, определяющая будет куча максимальной или минимальной.
        По умолчанию _compn=le, что означает минимальную кучу.
    """

    # Следуем по пути к корню, поднимая новый элемент наверх путём свапа с родителем, пока не найдем его место
    while pos > rootpos:
        parentpos = (pos - 1) // d
        if comp(heap[parentpos], heap[pos]):
            break   # Инвариант кучи теперь соблюдается

        # Меняем местами родителя с потомком, т.к. потомок приоритетней родителя
        heap[pos], heap[parentpos] = heap[parentpos], heap[pos]
        pos = parentpos


def _siftdown(heap: list[Any], d: int, pos: int, comp=lt):
    """Просеять вниз за время O(d*log(d, N))

    Алгоритм меняет местами объект в позиции pos с его потомками до тех пор, пока свойство кучи не восстановится.

    :param heap: вся куча в виде списка элементов
    :param d: коэффициент ветвления (максимальное число потомков у одного элемента)
    :param pos: индекс верхнего элемента в поддереве
    :param comp: функция сравнения двух элементов, определяющая будет куча максимальной или минимальной.
        По умолчанию _compn=le, что означает минимальную кучу.
    """

    # Следуем по пути к листьям, меняя элемент в позиции pos с его более приоритетным потомком
    while d*pos + 1 < len(heap):
        child = d*pos + 1   # Первый потомок
        # Находим самого приоритетного потомка
        for i in range(2, d+1):
            if d*pos + i >= len(heap):
                break
            if comp(heap[d*pos+i], heap[child]):
                child = d*pos + i

        if comp(heap[pos], heap[child]):
            break   # Свойство кучи соблюдено

        heap[pos], heap[child] = heap[child], heap[pos]
        pos = child


def _heapify(items: list[Any], d: int, comp=lt) -> list[Any]:
    """Создать на месте кучу из списка за время O(log(len(items))).

    :param items: список элементов
    :param d: коэффициент ветвления (максимальное число потомков у одного элемента)
    :param comp: функция сравнения двух элементов, определяющая будет куча максимальной или минимальной.
        По умолчанию _compn=le, что означает минимальную кучу.
    :return: куча из элементов items.

    """
    for i in range(len(items)):
        _siftup(items, d, 0, i, comp=comp)
    return items


def main():
    heap = DHeap(4)         # Пустая минимальная 4-куча
    heap.push(1)            # Добавление элемента 1 в кучу
    heap.push(2)            # Добавление элемента 2 в кучу
    heap.pop()              # Удаление наименьшего элемента
    print(heap)             # MinDHeap([2])
    heap.replace(3)         # Удаляет верхний элемент и добавляет новый
    print(heap)             # MinDHeap([3])
    heap.pushpop(4)         # эквивалентно последовательному вызову heap.push(4) и heap.pop(), но работает быстрее

    print(heap + MinDHeap(2, [1, 2, 3, 4, 5]))  # [1, 3, 2, 4, 4, 5] - список, хранящий элементы в виде кучи.
    # Того же результата можно достичь, просто прибавив к heap [1, 2, 3, 4, 5]

    # Дополнение кучи новыми элементами (можно прибавлять другую кучу, множество и т.д.)
    heap += [1, 2, 3, 4, 5]
    print(heap)  # MinDHeap([1, 3, 2, 4, 4, 5])

    # Слияние кучи со всеми наборами элементов из *args за линейное время
    heap.merge([0], {-1}, (-2, -3), MaxDHeap(1, [-4, -5]))
    print(heap)  # MinDHeap([-5, -3, -4, 0, -2, 1, 2, 4, 3, 4, -1, 5])

    # Вывести верхний элемент  кучи можно двумя способами
    print(heap[0], heap.head)   # ->  -5, -5
    print(repr(heap))
    print(MaxDHeap(1, [3]).replace(4))
    print(MaxDHeap(1, list(heap.items)))
    print(heap.children(1))
    print(MinDHeap(4, list(range(31))))
    print("___")
    print(heap)
    h = deepcopy(heap)
    h.push(999)
    print(h)
    print(heap)
    print((h := MinDHeap(2, [0, 1, 8, 9, 3, -9, -100, 5, 3, 99])) + [1000, -1000, 2000])
    print(h.head, h.tail)
    h.remove(-100)
    h.remove(500)
    h.remove(3)
    print(h)
    h.replace(10001)
    h += [10, 10]
    print(h.pop())
    print(h)
    print(h.pop(2))
    print(h)
    heap = DHeap(3, [0, -10, 8, 8, 6, 3, 3, 3, 24, 1, 2, -5, -6])
    print(heap)
    print(heap[1], heap[-3], heap[-1], heap.head, heap.tail)


if __name__ == "__main__":
    main()
