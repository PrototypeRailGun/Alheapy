import unittest
from alheapy.binomial_heap import BinomialHeap
from random import shuffle


data = list(range(-50, 51)) * 3
shuffle(data)


def get_inversions(heap: BinomialHeap):
    """Инверсией назовем пару родитель-потомок, в которой приоритет потомка выше родителя.

    :return: список пар (<пара индексов родителя и потомка>, <родитель, потомок>)
    """
    def traverse(node):
        child = node.child
        while child is not None:
            if child.value != node.value and comp(child, node):
                inversions.append((node, child))
            if child is not None:
                traverse(child)
            child = child.sibling

    inversions = []
    comp = heap._compn
    for root in heap._roots:
        traverse(root)
    return inversions


def assert_degrees(heap: BinomialHeap):
    wrong = []
    def traverse(node):
        while node is not None:
            if node.child is not None:
                if node.child.degree >= node.degree:
                    wrong.append((node, node.child))
                traverse(node.child)

            if node.sibling is not None and node.sibling.degree >= node.degree:
                wrong.append((node, node.sibling))
            node = node.sibling

    for i in range(len(heap._roots)):
        root = heap._roots[i]
        if i+1 < len(heap._roots):
            r = heap._roots[i + 1]
            if root.degree <= r.degree:
                wrong.append((root, r))
        traverse(root)
    return wrong


class TestBinomialHeap(unittest.TestCase):
    def test_init_head_size(self):
        heap = BinomialHeap()
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(assert_degrees(heap), [])
        self.assertEqual(heap.head, None)
        self.assertEqual(heap.size, 0)

        heap = BinomialHeap([0])
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(assert_degrees(heap), [])
        self.assertEqual(heap.head, 0)
        self.assertEqual(heap.size, 1)


        heap = BinomialHeap(data)
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(assert_degrees(heap), [])
        self.assertEqual(heap.head, -50)
        self.assertEqual(heap.size, 303)

    def test_merge(self):
        heap = BinomialHeap()
        heap.merge([], {}, (), BinomialHeap())
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(assert_degrees(heap), [])
        self.assertEqual(heap.head, None)
        self.assertEqual(heap.size, 0)

        heap.merge([1], BinomialHeap([0]))
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(assert_degrees(heap), [])
        self.assertEqual(heap.head, 0)
        self.assertEqual(heap.size, 2)

        heap.merge((), data, [])
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(assert_degrees(heap), [])
        self.assertEqual(heap.head, -50)
        self.assertEqual(heap.size, 305)

    def test_push(self):
        heap = BinomialHeap()
        heap.push(10)
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(assert_degrees(heap), [])
        self.assertEqual(heap.head, 10)
        self.assertEqual(heap.size, 1)
        heap.push(-10)
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(assert_degrees(heap), [])
        self.assertEqual(heap.head, -10)
        self.assertEqual(heap.size, 2)

        size = 2
        for i, val in enumerate(data):
            size += 1
            heap.push(val)
            self.assertEqual(heap.size, size)
            self.assertEqual(heap.head, min([-10, min(data[:i+1])]))

    def test_pop(self):
        heap = BinomialHeap()
        heap.push(10)
        top = heap.pop()
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(assert_degrees(heap), [])
        self.assertEqual(len(heap), 0)
        self.assertEqual(top, 10)
        self.assertEqual(heap.head, None)
        self.assertEqual(heap.size, 0)

        heap.merge(data)
        while heap:
            target = heap.head
            top = heap.pop()
            self.assertEqual(top, target)
            self.assertEqual(get_inversions(heap), [])
            self.assertEqual(assert_degrees(heap), [])

    def test_replace(self):
        heap = BinomialHeap()
        heap.push(10)
        top = heap.replace(5)
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(assert_degrees(heap), [])
        self.assertEqual(len(heap), 1)
        self.assertEqual(top, 10)
        self.assertEqual(heap.head, 5)
        self.assertEqual(heap.size, 1)

        heap.merge(data)
        for i in data:
            target = heap.head
            top = heap.replace(i)
            self.assertEqual(top, target)
            self.assertEqual(get_inversions(heap), [])
            self.assertEqual(assert_degrees(heap), [])

    def test_pushpop(self):
        heap = BinomialHeap()
        top = heap.pushpop(10)
        self.assertEqual(top, 10)
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(assert_degrees(heap), [])
        self.assertEqual(heap.size, 0)
        self.assertEqual(heap.head, None)

        heap.push(10)
        top = heap.pushpop(-10)
        self.assertEqual(top, -10)
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(assert_degrees(heap), [])
        self.assertEqual(heap.size, 1)
        self.assertEqual(heap.head, 10)

        heap = BinomialHeap([-10])
        top = heap.pushpop(5)
        self.assertEqual(top, -10)
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(assert_degrees(heap), [])
        self.assertEqual(heap.size, 1)
        self.assertEqual(heap.head, 5)

        heap.merge(data)
        top = heap.pushpop(1000)
        self.assertEqual(top, -50)
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(assert_degrees(heap), [])
        self.assertEqual(heap.head, -50)
        top = heap.pushpop(-1000)
        self.assertEqual(top, -1000)
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(assert_degrees(heap), [])
        self.assertEqual(heap.head, -50)

        for i in data:
            target = heap.head
            top = heap.pushpop(i)
            self.assertEqual(top, min(target, i))
            self.assertEqual(get_inversions(heap), [])
            self.assertEqual(assert_degrees(heap), [])

    def test_find(self):
        heap = BinomialHeap()
        self.assertEqual(heap.find(0), None)
        heap = BinomialHeap([1])
        self.assertEqual(heap.find(1).value, 1)
        self.assertEqual(heap.find(0), None)

        heap = BinomialHeap(data)
        for i in data:
            self.assertEqual(heap.find(i).value, i)

    def test_update(self):
        heap= BinomialHeap(data)
        for i in data:
            heap.update(i, i*100)
            self.assertEqual(get_inversions(heap), [])

    def test_remove(self):
        heap = BinomialHeap()
        heap.remove(0)
        self.assertEqual(heap.head, None)

        heap = BinomialHeap([1])
        heap.remove(0)
        self.assertEqual(heap.head, 1)
        heap.remove(1)
        self.assertEqual(heap.head, None)

        heap = BinomialHeap(data)
        for i in data:
            heap.remove(i)
            self.assertEqual(get_inversions(heap), [])
            self.assertEqual(assert_degrees(heap), [])


if __name__ == "__main__":
    unittest.main()
