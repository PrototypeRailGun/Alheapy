# -*- coding: utf-8 -*-
import unittest
from typing import Tuple
from ..beap import Beap, MinBeap, MaxBeap
from random import shuffle, choice, randint


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


data = list(range(10000)) * 3
shuffle(data)


class TestInit(unittest.TestCase):
    def test_init_1(self):
        self.assertEqual(Beap()._items, [])

    def test_init_2(self):
        self.assertEqual(Beap([7, 3, 0, -5], ordered=True)._items, [7, 3, 0, -5])

    def test_init_3(self):
        self.assertEqual(Beap([1, 2, 3, 11, 12, 13], ordered=True)._items, [1, 2, 3, 11, 12, 13])

    def test_init_4(self):
        self.assertEqual(Beap([])._items, [])

    def test_init_5(self):
        self.assertEqual(get_inversions(Beap([7, 3, 0, -5])), [])


class TestTopAndTail(unittest.TestCase):
    def test_0_elements(self):
        beap = Beap()
        self.assertEqual(beap.head, None)
        self.assertEqual(beap.tail, None)

    def test_1_element(self):
        beap = Beap([3])
        self.assertEqual(beap.head, 3)
        self.assertEqual(beap.tail, 3)

    def test_2_elements(self):
        beap = Beap([1, 2])
        self.assertEqual(beap.head, 1)
        self.assertEqual(beap.tail, 2)

    def test_many_elements(self):
        beap = Beap([0, 6, 12, 0, 6, 2, 3, -1, -100, -100])
        self.assertEqual(beap.head, -100)
        self.assertEqual(beap.tail, 12)


class TestPush(unittest.TestCase):
    def test_push(self):
        beap = Beap([0, 3, 7, 9, -1, 7, -30, -765, 1, 89, 456])
        print(beap, beap.height)
        self.assertEqual(get_inversions(beap), [])
        beap.push(999)
        beap.push(-999)
        self.assertEqual(get_inversions(beap), [])


class TestPop(unittest.TestCase):
    def test_pop_when_1_element(self):
        beap = Beap([3])
        p = beap.pop()
        self.assertEqual(p, 3)

    def test_pop_when_2_elements(self):
        beap = Beap([1, 2])
        p = beap.pop()
        self.assertEqual(p, 1)
        p = beap.pop()
        self.assertEqual(p, 2)

    def test_pop(self):
        beap = Beap([0, 3, 7, 9, -1, 7, -30, -765, 1, 89, 456])
        self.assertEqual(get_inversions(beap), [])
        beap.pop()
        self.assertEqual(get_inversions(beap), [])
        beap.pop()
        self.assertEqual(get_inversions(beap), [])
        print(beap)
        for _ in range(9):
            i = randint(-len(beap), len(beap)-1)
            print(i, beap.pop(i))
            print(beap)
            self.assertEqual(get_inversions(beap), [])


class TestReplace(unittest.TestCase):
    def test_replace_when_1_element(self):
        beap = Beap([3])
        p = beap.replace(1)
        self.assertEqual(p, 3)
        self.assertEqual(get_inversions(beap), [])

    def test_replace_when_2_elements(self):
        beap = Beap([1, 2])
        p = beap.replace(3)
        self.assertEqual(p, 1)
        self.assertEqual(get_inversions(beap), [])

    def test_replace_greatest_and_smallest(self):
        beap = Beap([0, 3, 7, 9, -1, 7, -30, -765, 1, 89, 456])
        self.assertEqual(get_inversions(beap), [])
        beap.replace(-1000)   # Новый элемент больше всех остальных (по приоритету)
        self.assertEqual(get_inversions(beap), [])
        beap.replace(2000)   # Новый элемент меньше всех остальных (по приоритету)
        self.assertEqual(get_inversions(beap), [])

    def test_replace_sith_random_idx(self):
        beap = Beap([0, 3, 7, 9, -1, 7, -30, -765, 1, 89, 456])
        for _ in range(len(beap)):
            i = randint(-len(beap), len(beap)-1)
            val = randint(-100, 100)
            beap.replace(val, i)
            self.assertEqual(get_inversions(beap), [])


class TestPushpop(unittest.TestCase):
    def test_pushpop_when_1_element(self):
        beap = Beap([3])
        p = beap.pushpop(1)
        self.assertEqual(p, 1)
        self.assertEqual(get_inversions(beap), [])

    def test_pushpop_when_2_elements(self):
        beap = Beap([1, 2])
        p = beap.pushpop(3)
        self.assertEqual(p, 1)
        self.assertEqual(get_inversions(beap), [])

    def test_pushpop_greatest_smallest_ordinary(self):
        beap = Beap([0, 3, 7, 9, -1, 7, -30, -765, 1, 89, 456])
        self.assertEqual(get_inversions(beap), [])
        p = beap.pushpop(-1000)   # Новый элемент больше всех остальных (по приоритету)
        self.assertEqual(p, -1000)
        self.assertEqual(get_inversions(beap), [])
        p = beap.pushpop(1000)   # Новый элемент меньше всех остальных (по приоритету)
        self.assertEqual(p, -765)
        self.assertEqual(get_inversions(beap), [])
        p = beap.pushpop(4)
        self.assertEqual(p, -30)   # Новый элемент ни максимальный, ни минимальный (обычный случай)
        self.assertEqual(get_inversions(beap), [])


class TestFind(unittest.TestCase):
    def empty_beap(self):
        beap = Beap()
        idx = beap.find(1)
        self.assertEqual(idx, -1)

    def one_element_in_the_beap(self):
        beap = Beap([1])
        idx = beap.find(1)
        self.assertEqual(idx, 0)
        idx = beap.find(7)   # Поиск несуществующего элемента
        self.assertEqual(idx, -1)

    def many_elements_in_the_beap(self):
        items = [0, 3, 7, 9, -1, 7, -30, -765, 1, 89, 456]
        beap = Beap(items)
        indices = [4, 3, 8, 6, 1, 8, 2, 0, 5, 9, 10]
        for i in set(items):
            self.assertEqual(beap.find(i), indices[i])


class TestUpdate(unittest.TestCase):
    def test(self):
        beap = Beap([0, 3, 7, 9, -1, 7, -30, -765, 1, 89, 456])
        for _ in range(100):
            old = choice(beap._items)
            new = randint(-100, 100)
            beap.update(old, new)
            self.assertEqual(get_inversions(beap), [])


class TestRemove(unittest.TestCase):
    def test(self):
        beap = Beap([0, 3, 7, 9, -1, 7, -30, -765, 1, 89, 456])
        for _ in range(len(beap)):
            val = choice(beap._items)
            beap.remove(val)
            self.assertEqual(get_inversions(beap), [])


class TestIadd(unittest.TestCase):
    def test(self):
        beap = Beap([0, 3, 7, 9, -1, 7, -30, -765, 1, 89, 456])
        beap += [57, -9, 13, 999]
        self.assertEqual(get_inversions(beap), [])


class TestMerge(unittest.TestCase):
    def test(self):
        beap = Beap([0, 3, 7, 9, -1, 7, -30, -765, 1, 89, 456])
        beap.merge([0, 9], (), set(), {45, 77, 0}, [0, 1, 5, 87])
        self.assertEqual(get_inversions(beap), [])


class TestLayer(unittest.TestCase):
    def test(self):
        beap = Beap([0, 3, 7, 9, -1, 7, -30, -765, 1, 89, 456])
        self.assertEqual(set(beap.layer(1)), {-765})
        self.assertEqual(set(beap.layer(2)), {-30, -1})
        self.assertEqual(set(beap.layer(3)), {3, 0, 1})
        self.assertEqual(set(beap.layer(4)), {7, 9, 7, 89})
        self.assertEqual(set(beap.layer(5)), {456})


class BigTest(unittest.TestCase):
    def test(self):
        beap = Beap(data)
        self.assertEqual(get_inversions(beap), [])


class TestMinBeap(unittest.TestCase):
    def test(self):
        beap = MinBeap(data)
        self.assertEqual(get_inversions(beap), [])


class TestMaxBeap(unittest.TestCase):
    def test(self):
        beap = MaxBeap(data)
        self.assertEqual(get_inversions(beap), [])


data1 = list(range(-100, 100)) * 200
shuffle(data1)
data2 = list(range(-100, 100)) * 200
shuffle(data2)


if __name__ == "__main__":
    unittest.main()
