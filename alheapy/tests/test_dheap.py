import unittest
from alheapy.dheap import DHeap
from random import shuffle, randint
from copy import copy


data = list(range(-5, 10)) * 3
shuffle(data)


def get_inversions(heap):
    inversions = []
    for i in range(1, len(heap)):
        if heap[i] != heap[(i-1)//heap.d] and heap.comp(heap[i], heap[(i - 1) // heap.d]):
            inversions.append((i, (i-1)//heap.d))
    return inversions


class TestDHeap(unittest.TestCase):
    def test_build(self):
        heap = DHeap(3, copy(data))
        self.assertEqual(get_inversions(heap), [])

    def test_push(self):
        heap = DHeap(3, copy(data))
        heap.push(100)
        self.assertEqual(get_inversions(heap), [])
        heap.push(-100)
        self.assertEqual(get_inversions(heap), [])
        heap.push(5)
        self.assertEqual(get_inversions(heap), [])

    def test_merge(self):
        heap = DHeap(3, copy(data))
        self.assertEqual(get_inversions(heap + [1, 2, 3]), [])
        heap += [10, 20, 30]
        self.assertEqual(get_inversions(heap), [])
        heap.merge([40, 50], (), {60}, DHeap(1, [70, 80]))
        self.assertEqual(get_inversions(heap), [])

    def test_pop(self):
        heap = DHeap(3, copy(data))
        top = heap.pop()
        print(top, heap, get_inversions(heap))
        self.assertEqual(get_inversions(heap), [])
        top = heap.pop(5)
        print(top, heap)
        self.assertEqual(get_inversions(heap), [])
        top = heap.pop(-2)
        print(top, heap)
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(get_inversions(heap), [])
        top = heap.pop(-1)
        print(top, heap)
        top = heap.pop(-10)
        print(top, heap)
        self.assertEqual(get_inversions(heap), [])

    def test_replace(self):
        heap = DHeap(3, copy(data))
        top = heap.replace(-10)
        self.assertEqual(top, -5)
        self.assertEqual(get_inversions(heap), [])
        top = heap.replace(10)
        self.assertEqual(top, -10)
        self.assertEqual(get_inversions(heap), [])
        top = heap.replace(2)
        self.assertEqual(top, -5)
        self.assertEqual(get_inversions(heap), [])
        for _ in range(5):
            i = randint(-len(heap), len(heap)-1)
            val = randint(-1000, 1000)
            heap.replace(val, i)
            self.assertEqual(get_inversions(heap), [])

    def test_pushpop(self):
        heap = DHeap(3, copy(data))
        top = heap.pushpop(-10)
        self.assertEqual(top, -10)
        self.assertEqual(get_inversions(heap), [])
        top = heap.pushpop(10)
        self.assertEqual(top, -5)
        self.assertEqual(get_inversions(heap), [])
        top = heap.pushpop(2)
        self.assertEqual(top, -5)
        self.assertEqual(get_inversions(heap), [])

    def test_remove_and_del(self):
        heap = DHeap(3, copy(data))
        for _ in range(3):
            heap.remove(0)
        self.assertEqual(set(heap.items) & {0}, set())
        heap.remove(3)
        heap.remove(5)
        heap.remove(0)
        del heap[0]
        del heap[-3]
        self.assertEqual(get_inversions(heap), [])

    def test_update(self):
        heap = DHeap(3, copy(data))
        for _ in range(3):
            i = heap.find(-5)
            heap.update(i, -50)
        print(heap)
        self.assertEqual(get_inversions(heap), [])
        for _ in range(3):
            heap.update(-1, 100)
        print(heap)
        self.assertEqual(get_inversions(heap), [])
        heap[0], heap[10], heap[20] = 0, 1000, -20000
        print(heap)
        self.assertEqual(get_inversions(heap), [])


if __name__ == "__main__":
    unittest.main()
