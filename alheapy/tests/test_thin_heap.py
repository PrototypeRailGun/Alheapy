import unittest
from alheapy.thin_heap import ThinHeap
from random import shuffle


data = list(range(-50, 51)) * 3
shuffle(data)


def get_inversions(heap: ThinHeap):
    """Инверсией назовем пару родитель-потомок, в которой приоритет потомка выше родителя.

    :return: список пар (<пара индексов родителя и потомка>, <родитель, потомок>)
    """
    def traverse(node):
        if node is None:
            return

        right = node
        while right is not None:
            if right.child and comp(right.child, right):
                inversions.append((right, right.child))
            traverse(right.child)
            right = right.right

    inversions = []
    comp = heap._compn
    traverse(heap._head)

    return inversions


class TestThinHeap(unittest.TestCase):
    def test_push(self):
        heap = ThinHeap()
        top, size = None, 0
        for i, x in enumerate(data):
            heap.push(x)
            size += 1
            top = min(data[:i+1])
            self.assertEqual(heap.size, size)
            self.assertEqual(heap.head, top)

    def test_merge(self):
        heap = ThinHeap()
        heap.merge(data, ThinHeap(data[:10]), (), [1, 2, 3], set())
        self.assertEqual(heap.size, 316)
        self.assertEqual(heap.head, -50)


    def test_pop(self):
        heap = ThinHeap([0])
        self.assertEqual(heap.pop(), 0)
        self.assertEqual(heap.head, None)
        self.assertEqual(heap.size, 0)

        heap = ThinHeap(data)
        size = 303
        while heap:
            top = heap.head
            self.assertEqual(heap.pop(), top)
            size -= 1
            self.assertEqual(heap.size, size)
            self.assertEqual(get_inversions(heap), [])

    def test_find(self):
        heap = ThinHeap()
        self.assertEqual(heap.find(0), None)
        heap.push(0)
        self.assertEqual(heap.find(0).value, 0)
        self.assertEqual(heap.find(1), None)

        heap.merge(data)
        for i in data:
            self.assertEqual(heap.find(i).value, i)
        self.assertEqual(heap.find(999), None)
        heap.pop()
        for i in data:
            self.assertEqual(heap.find(i).value, i)
        self.assertEqual(heap.find(999), None)

    def test_replace(self):
        heap = ThinHeap([0])
        self.assertEqual(heap.replace(1), 0)
        self.assertEqual(heap.head, 1)
        self.assertEqual(heap.size, 1)

        heap.merge(data)
        for i in data + [999, -999]:
            top = heap.head
            self.assertEqual(heap.replace(i), top)
            self.assertEqual(get_inversions(heap), [])

    def test_pushpop(self):
        heap = ThinHeap([0])
        self.assertEqual(heap.pushpop(1), 0)
        self.assertEqual(heap.pushpop(-1), -1)
        self.assertEqual(heap.head, 1)
        self.assertEqual(heap.size, 1)

        heap.merge(data)
        for i in data + [999, -999]:
            top = heap.head
            self.assertEqual(heap.pushpop(i), min(i, top))
            self.assertEqual(get_inversions(heap), [])


if __name__ == "__main__":
    unittest.main()
