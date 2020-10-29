# -*- coding: utf-8 -*-
"""Bi-parental heap (Beap)

Двуродительская куча (англ. bi-parental heap или beap) — такая куча,
где у каждого элемента обычно есть два ребенка (если это не последний уровень)
и два родителя (если это не первый уровень).

Вот пример beap, содержащей элементы от 0 до 9
          0            # Слой (уровень) 1
        1   2          # Уровень 2
      3   4   5        # Уровень 3
    6   7   8   9      # Уроень 4

Пусть N - число элементов в двуродительской куче (beap).
Как и любая куча, beap удовлетворяет двум свойствам (invariant):
1) Является полным бинарным деревом (только последний слой может быть незаполнен)
2) Приоритет потомка меньше или равен приоритета каждого из родителей.
Высота beap составляет приблизительно sqrt(2N), а если в целых числах, то round(sqrt(2N)).

В данной реализации нумерация элементов идет слева направо по слоям и начинается с 0,
нумерация уровней идет сверху вниз и начинается с 1.
Элементы кучи хранятся в списке послойно и без разделителей, например [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].

Для элемента i (0-based) в слое k левым родителем (если такой существует) будет элемент i - k,
а правым (если существует) i - k + 1.
Для полного слоя k индексами его начального и конецного элементов будут соотв. k * (k - 1)//2 и k * (k + 1)//2 - 1

----------------------------------------------------------------
 № | Базовые поддерживаемые операции           |   Сложность   |
---|-------------------------------------------|---------------|
 1 | Получение верхнего                        | O(1)          |
   |(самого приоритетного элемента)            |               |
---|------------------------------------------------------------
 2 | Получение минимального по приоритету      | O(sqrt(2N))   |
   | элемента                                  |               |
----------------------------------------------------------------
 3 | Вставка                                   | O(sqrt(2N))   |
----------------------------------------------------------------
 4 | Удаление верхнего элемента                | O(sqrt(2N))   |
----------------------------------------------------------------
 5 | Поиск                                     | O(sqrt(2N))   |
----------------------------------------------------------------
 6 | Построение beap                           | O(N*log(N))   |
----------------------------------------------------------------

Детали реализации вы найдете в коде. Примеры использования в main().
Ссылка на авторскую статью: http://acm.math.spbu.ru/~sk1/download/books/heaps/Beap.pdf
"""

__all__ = ["Beap", "MinBeap", "MaxBeap"]


from math import sqrt
from operator import lt, gt
from copy import copy, deepcopy
from typing import Any, Tuple, Callable
from collections.abc import Iterable

from alheapy._heap import Heap, HeapIndexError


class Beap(Heap):
    """Bi-parental heap

    # Основные сведения:
    1) Beap(values) является оберткой для входных данных, поэтому любые изменения в куче
       затрагивают исходный список.
    2) При инициализации можно указать параметр _compn (по умолчанию =operator.lt), этот
       эта функция нужна для сравнения двух элементов:
       (True если приоритет a больше (или равен), чем у b и False в противном случае)
    3) Отсортированный список удовлетворяет инварианту beap, поэтому если вы можете упорядочить
       данные перед инициализацией кучи, действительно стоит сделать это, ведь встроенная сортировка
       посторет beap на порядок быстрее.
    4) Если вы передаете заранее упорядоченные данные, то укажите параметр ordered=True: Beap(values, ordered=True)

    beap = Beap([...])

    # Публичные атрибуты и @property:
    self.height - высота beap (количество слоев)
    beap.head - верхний (самый приоритетный элемент)
    beap.tail - элемент с наименьшим приоритетом
    beap.items - список элементов кучи

    # Cтатические методы:
    beap.span(i) - кортеж (start, end) - начальный и конечный индексы слоя i

    # Публичные методы:
    beap.push(item) - добавление нового элемента
    beap.pop(idx=0) - удалить и вернуть элемент с индесом idx, по умолчанию idx=0 - верхний элемент
    beap.replace(item, idx=0) - удалить и вернуть элемент с индексом idx, а затем вставить на его место новый
    beap.pushpop(item) - эквивалентно последовательному вызову beap.push() и beap.pop(),
        но использует лишь одну операцию просевивания, что повышает эффективность почти в 2 раза.
    beap.find(val) - с учетом val, найти номер элемента с таким значением и его уровень или вернуть (-1, -1)
        в случае отсутсвия
    beap.update(old, new) - заменить первое вхождение элемента new элементом old и вернуть индекс old
    beap.remove(val) - удалить первый найденный элемент со значением val
    beap.merge(*args) - добавление в кучу элементов из *args
    beap.is_beap() - соблюдается ли инвариант (функция только для тестирования и отладки)
    beap.layer(i) - элементы уровня i (срез внутреннего списка)
    beap.comp(a, b) - True если приоритет a больше (или равен), чем у b и False в противном случае

    """
    def __init__(self, values: Iterable = None, ordered: bool = False, comp: Callable[[Any, Any], bool] = lt):
        """
        :param values: список элементов для объединения в beap
        :param ordered: упорядочены ли входные данные
        :param comp: функция, принимающая 2 аргумента и возвращающая True, если
            приоритет первого больше (или равен) приоритету второго.
        """
        self._compfunc = comp

        self._items = values
        if self._items is None:
            self._items = []
        if not isinstance(self._items, list):
            self._items = list(self._items)

        if ordered:
            # Данные упорядочены => свойство beap соблюдено
            self.height = round(sqrt(2 * len(self._items)))
        else:
            # Просеиваем вверх каждый элемент, выстраивая beap
            self.height = 0
            count = 0
            for i in range(len(self._items)):
                count += 1
                if count >= self.height:
                    self.height += 1
                    count = 0
                self._siftup(i, self.height)

    @staticmethod
    def span(i: int) -> (int, int):
        """Начальный и конечный индексы уровня i"""
        return i*(i-1)//2, i*(i+1)//2 - 1

    def is_beap(self) -> bool:
        """Соблюдается ли инвариант beap"""
        return not get_inversions(self)

    @property
    def head(self) -> Any:
        """Верхний элемент"""
        return self._items[0] if self._items else None

    @property
    def tail(self) -> Any:
        """Элемент с наименьшим приоритетом"""
        if not self._items:
            return None

        start, end = Beap.span(self.height)
        tail = self._items[start]
        for i in range(start+1, min(end+1, len(self._items))):
            if self.comp(tail, self._items[i]):
                tail = self._items[i]

        return tail

    def layer(self, i):
        """Элементы уроня i (нумерация уровней начинается с 1)"""
        start, end = Beap.span(i)
        return self._items[start:end+1]

    @property
    def items(self) -> list[Any]:
        """Элементы кучи"""
        return self._items

    def comp(self, first: Any, second: Any) -> bool:
        """True если приоритет first больше (или равен) приоритета second, и False иначе"""
        return self._compfunc(first, second)

    def push(self, item: Any):
        """Добавление нового элемента за O(sqrt(2N))"""

        # Обновляем высоту дерева
        start, end = Beap.span(self.height)
        if len(self._items) - 1 == end:
            self.height += 1

        self._items.append(item)
        self._siftup(len(self._items)-1, self.height)

    def pop(self, idx=0) -> Any:
        """Удаление элемента с индексом idx, по умолчанию idx=0 - верхний элемент

        Время работы: O(sqrt(2N)), где N - количество элементов в куче
        """
        if not isinstance(idx, int):
            raise TypeError("heap list index must be integers, not {}".format(type(idx)))

        if idx < 0:
            idx = len(self._items) + idx
        if len(self._items) <= idx or idx < 0:
            raise HeapIndexError("heap index out of range")

        last = self._items.pop()   # Вызовет IndexError, если куча пустая

        start, end = Beap.span(self.height)
        if len(self._items) == start:
            self.height -= 1

        if idx != len(self._items) and self._items:
            to_return = self._items[idx]
            # Ставим последний элемент первым, а потом просеиваем его вниз, восстанавливая свойства кучи
            self._items[idx] = last
            self._repair(idx)
            return to_return
        return last

    def replace(self, item: Any, idx=0) -> Any:
        """Замена элемнта в позиции idx элементом item.

        По умолчанию idx=0, в таком случае заменен и возвращен будет верхний элемент кучи.
        Работает за время O(sqrt(2N)), но быстрее, чем pop с последующим push,
        т.к. вызывает процедуру просеивания лишь один раз.

        """
        if not isinstance(idx, int):
            raise TypeError("heap list index must be integers, not {}".format(type(idx)))

        if idx < 0:
            idx = len(self._items) + idx
        if len(self._items) <= idx or idx < 0:
            raise HeapIndexError("heap index out of range")

        to_return = self._items[0]

        # Ставим новый элемент на место удаляемого и восстанавливаем инвариант кучи
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
            self._siftdown(0, 1)
        return item

    def find(self, val: Any) -> int:
        """Поиск элемента

        С учетом значения val, найти индекс элемента с таким значением и его уровень или вернуть -1, -1,
        если такого элемента не существует.
        Время работы не превышает O(sqrt(2N)).
        """

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Пусть есть beap         0
        #                       1   2
        #                     3   4   5
        #                   6   7   8   9
        # Рассмотрим её как левый верхний угол матрицы:
        #
        #   0 2 5 9
        #   1 4 8
        #   3 7
        #   6
        #
        # Начнем поиск с правого верхнего угла (последний элемент списка self._items)
        # 1) Если приоритет искомого элемента больше, чем у элемента в текущей позиции, то переходим влево по строке
        # 2) Если приоритет искомого элемента меньше, чем у элемента в текущей позиции, то перемещаемя вниз по столбцу,
        # 3) а если внизу элемента нет, то перемещаемся вниз и влево (=влево по последнему слою кучи).
        # 4) Как только находим элемент с равным val приоритетом, вызвращаем его индекс, а если мы оказываемся в левом
        # нижнем углу и значение в нем не равно val, значит, искомого элемента не существует и пора вернуть -1.
        #
        # В приведенной в примере кучи захотим найти элемент 3. Начнем с элемента 9. Т.к. куча минимальная, то
        # приоритет 3 больше приоритета 9, перейдем влево. Теперь текущий элемент - 5. На втором шаге убедимся,
        # что приоритет искомого элемента всё еще выше текущего и снова перейдем влево - в позицию элемента 2.
        # Т.к. приоритет 2 выше 3, спустимся вниз к элементу 4, затем влево -> 1 и вниз -> 3. Элемент val=3 найден.
        #
        # Примечание: если последний слой кучи не полон, то левым верхним углом будем считать последний элемент
        # предпоследнего слоя.
        #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        if not self._items:
            return -1

        level = self.height
        lower_left, right_up = Beap.span(level)
        if right_up >= len(self._items):
            level -= 1
            _, right_up = Beap.span(level)

        pos = right_up
        while pos != lower_left:
            if self._items[pos] == val:
                return pos

            start, end = Beap.span(level)
            level_pos = pos - start

            if self.comp(val, self._items[pos]):
                # Случай 1: переходим влево
                prev_start, _ = Beap.span(level - 1)
                pos = prev_start + level_pos - 1
                level -= 1

            elif self.comp(self._items[pos], val) and level < self.height:
                next_start, next_end = Beap.span(level + 1)
                if next_start + level_pos >= len(self._items):
                    # Случай 3: переходим лево и вниз (по диагонали)
                    pos -= 1
                else:   # Случай 2: переходим вниз
                    pos = next_start + level_pos
                    level += 1

            else:   # Тоже случай 3
                pos -= 1

        return lower_left if val == self._items[lower_left] else -1

    def update(self, old: Any, new: Any):
        """Заменить первый найденный элемент old элементом new"""
        if new == old:
            return
        idx = self.find(old)
        if idx != -1:
            self._items[idx] = new
            self._repair(idx)

    def remove(self, val: Any):
        """Удаление первого найденного элемента со значением val

        Сложность O(sqrt(2N))
        """
        idx = self.find(val)
        if idx != -1:
            self.pop(idx)

    def merge(self, *args: Iterable):
        """Добавление в beap элементов из *args"""
        for arg in args:
            for i in arg:
                self.push(i)

    def _repair(self, pos):
        """Восстановить инвариант beap"""
        if pos == 0:
            self._siftdown(pos, 1)
        else:
            level = round(sqrt(2 * (pos+1)))
            self._siftup(pos, level)
            self._siftdown(pos, level)

    def _siftup(self, pos: int, level):
        """Просеять вверх за время O(sqrt(2N))"""

        # Меняем текущий элемент с его наименее приоритетным родителем до тех пор
        # Пока свойство beap не будет восстановлено
        start, end = Beap.span(level)
        while level:
            level_pos = pos - start   # Позиция элемента на уровне
            prev_start, prev_end = Beap.span(level-1)   # Начало и конец предыдущего уровня

            # Левый и правый родители
            if level_pos > 0:
                left_parent = prev_start + level_pos - 1
                if level_pos == level - 1:
                    # Правого родителя нет, поэтому наименее приоритетным родителем является левый
                    parent = prev_end

                elif self.comp(self._items[left_parent], self._items[prev_start+level_pos]):
                    parent = prev_start + level_pos   # Приоритет правого родителя меньше
                else:
                    parent = prev_start + level_pos - 1   # Приоритет левого родителя меньше
            else:
                # Левого родителя нет, поэтому наименее приоритетным родителем является правый
                parent = prev_start

            if self.comp(self._items[parent], self._items[pos]):
                break   # Свойство beap соблюдено

            self._items[pos], self._items[parent] = self._items[parent], self._items[pos]
            pos = parent
            start, end = prev_start, prev_end
            level -= 1

    def _siftdown(self, pos, level):
        """Просеять вниз за время O(sqrt(2N))"""
        start, end = Beap.span(level)
        while level < self.height:
            next_start, next_end = Beap.span(level + 1)
            level_pos = pos - start

            # Находим наиболее приоритетного потомка
            child = next_start + level_pos
            if child >= len(self._items):
                break   # У элемента pos нет потомков

            if child + 1 < len(self._items) and self.comp(self._items[child+1], self._items[child]):
                child += 1

            if self.comp(self._items[pos], self._items[child]):
                break   # Свойство beap соблюдено

            self._items[pos], self._items[child] = self._items[child], self._items[pos]
            pos = child
            level += 1
            start, end = next_start, next_end

    def __add__(self, other: Iterable):
        new = copy(self)
        new.merge(other)
        return new

    def __iadd__(self, other: Iterable):
        self.merge(other)
        return self

    def __str__(self):
        return "Beap({})".format(self._items)

    def __repr__(self):
        return str(self.items)

    def __bool__(self):
        return len(self._items) > 0

    def __contains__(self, item: Any):
        return self.find(item) != -1

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx: int):
        """Элемент под номером idx в списке элементов кучи."""
        if not isinstance(idx, int):
            raise TypeError("heap list index must be integers, not {}".format(type(idx)))
        elif len(self._items) <= idx:
            raise HeapIndexError("heap list index out of range")
        return self._items[idx]

    def __setitem__(self, idx: int, value: Any):
        self.replace(value, idx)

    def __delitem__(self, idx):
        self.pop(idx)

    def __iter__(self):
        return iter(self._items)

    def __copy__(self):
        return Beap(copy(self._items), comp=self._compfunc)

    def __deepcopy__(self, memo):
        return Beap(deepcopy(self._items), comp=self._compfunc)


class MinBeap(Beap):
    """Класс минимальной версии Beap

    Верхним элементом будет элемент с самым малым значением.
    Прежде чем использовать MinBeap(), убедитесь, что передаваемые конструктору данные
    подлежат сортировке встроенной функцией sorted().
    В случае, если передаваемый список уже упорядочен, укажите ordered=True.

    """
    def __init__(self, values: Iterable, ordered: bool = False):
        if not ordered and values is not None:
            if not isinstance(values, list):
                values = list(values)
            values = sorted(values)
        super(MinBeap, self).__init__(values=values, ordered=True, comp=lt)

    def __str__(self):
        return "MinBeap({})".format(self._items)


class MaxBeap(Beap):
    """Класс максимальной версии Beap

    Верхним элементом будет элемент с самым большим значением.
    Прежде чем использовать MinBeap(), убедитесь, что передаваемые конструктору данные
    подлежат сортировке встроенной функцией sorted().
    В случае, если передаваемый список уже упорядочен, укажите ordered=True.

    """
    def __init__(self, values: Iterable, ordered: bool = False):
        if not ordered and values is not None:
            if not isinstance(values, list):
                values = list(values)
            values = sorted(values, reverse=True)
        super(MaxBeap, self).__init__(values=values, ordered=True, comp=gt)

    def __str__(self):
        return "MaxBeap({})".format(self._items)


def get_inversions(beap: Beap) -> list[Tuple[int, int]]:
    """Инверсией назовём пару потомок-родитель, в которой приоритет потомка выше, чем у родителя.
    :return: список инверсий.
    """
    inversions = []

    level = beap.height
    start, end = Beap.span(level)
    while level:
        prev_start, prev_end = Beap.span(level-1)

        # Сравниваем каждый элемент текущего уровня с левым родителем
        for i in range(1, level):
            if start + i >= len(beap):
                break
            if beap.comp(beap[start+i], beap[prev_start+i-1]):
                inversions.append((start+i, prev_start+i-1))

        # Сравниваем каждый элемент текущего уровня с правым родителем
        for i in range(level-1):
            if start + i >= len(beap):
                break
            if beap.comp(beap[start+i], beap[prev_start+i]):
                inversions.append((start+i, prev_start+i))

        start, end = prev_start, prev_end
        level -= 1

    return inversions


def main():
    """Примеры"""
    values1 = [0, 1, -1, -5, 2, 3, 4]
    values2 = [0, 1, -1, -5, 2, 3, 4]
    minb = MinBeap(values1)
    print(minb)
    maxb = Beap(values2, comp=gt)   # MaxBeap(values2) сделает то же, только быстрее
    print(maxb)

    minb.push(2)
    minb.push(-10)           # Новый head
    minb.push(10)            # Новый tail
    print(minb)
    top = minb.pop()         # Верхний элемент
    print(top, minb)
    top = minb.replace(11)   # Замена элемента -5 элементом 11
    print(top, minb)
    top = minb.pushpop(-100)
    print(top, minb)
    top = minb.pushpop(3)
    print(top, minb)
    print(minb.find(11))
    print(minb.find(200))
    print(minb.find(0), minb.find(1))
    minb.items[6] = 100
    print(minb.find(100))
    minb.merge(list(range(-10, -5)), {100, 200, 300})
    minb += maxb
    print(minb)
    print(get_inversions(minb))   # Инвариант не нарушен
    print(maxb)   # maxb не изменилась после minb += maxb
    print(maxb + [100, 200, 300], maxb)   # Результатом сложения стала новая куча, а исходная не изменилась
    print(maxb)
    maxb.pop(-3)
    print(maxb)


if __name__ == "__main__":
    main()
