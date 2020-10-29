import unittest
from alheapy.leftist_heap import LeftistHeap
from random import shuffle, choice


data = list(range(-50, 51)) * 3
shuffle(data)


def get_inversions(heap: LeftistHeap):
    """Инверсией назовем пару родитель-потомок, в которой приоритет потомка выше родителя.

    :return: список пар (<пара индексов родителя и потомка>, <родитель, потомок>)
    """
    def dfs(node):
        if node.parent is not None and comp(node, node.parent):
            inversions.append((node.parent, node))
        if node.left is not None:
            dfs(node.left)
        if node.right is not None:
            dfs(node.right)

    inversions = []
    comp = heap._compn
    if heap.head is not None:
        dfs(heap._head)
    return inversions


def get_violations(heap: LeftistHeap):
    """Соблюдается ли условие левосторонности"""
    # :return: список узлов, у которых left.dist < right.dist
    def dfs(node):
        if node.left is None and node.right is not None:
            violations.append(node)
        elif node.left is not None and node.right is not None:
            if node.left.dist < node.right.dist:
                violations.append(node)
        if node.left is not None:
            dfs(node.left)
        if node.right is not None:
            dfs(node.right)

    violations = []
    if heap.head is not None:
        dfs(heap._head)
    return violations


class TestLeftistHeap(unittest.TestCase):
    def test_init_head_size(self):
        heap = LeftistHeap()
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(get_violations(heap), [])
        self.assertEqual(heap.head, None)
        self.assertEqual(heap.size, 0)

        heap = LeftistHeap([])
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(get_violations(heap), [])
        self.assertEqual(heap.head, None)
        self.assertEqual(heap.size, 0)

        heap = LeftistHeap([1])
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(get_violations(heap), [])
        self.assertEqual(heap.head, 1)
        self.assertEqual(heap.size, 1)

        heap = LeftistHeap([1, 3, 5, 2, 4])
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(get_violations(heap), [])
        self.assertEqual(heap.head, 1)
        self.assertEqual(heap.size, 5)

        heap = LeftistHeap(iter(data))
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(get_violations(heap), [])
        self.assertEqual(heap.head, -50)
        self.assertEqual(heap.size, 303)

    def test_merge(self):
        heap = LeftistHeap(data)
        heap.merge([-1000, 1000])   # Новые минимум и максимум
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(get_violations(heap), [])
        self.assertEqual(heap.head, -1000)
        self.assertEqual(heap.size, 305)

        heap.merge([], [-4], (3, 5), {4, 5, 6}, {}, LeftistHeap([-1, -2, -3]))
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(get_violations(heap), [])
        self.assertEqual(heap.head, -1000)
        self.assertEqual(heap.size, 314)

        heap += data
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(get_violations(heap), [])
        self.assertEqual(heap.head, -1000)
        self.assertEqual(heap.size, 617)

    def test_push(self):
        heap = LeftistHeap()
        heap.push(1)
        heap.push(-1)
        heap.push(3)
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(get_violations(heap), [])
        self.assertEqual(heap.head, -1)
        self.assertEqual(heap.size, 3)

        size = 3
        for i in data:
            heap.push(i)
            size += 1
            self.assertEqual(get_inversions(heap), [])
            self.assertEqual(get_violations(heap), [])
            self.assertEqual(heap.size, size)

    def test_pop(self):
        heap = LeftistHeap([1])
        top = heap.pop()
        self.assertEqual(top, 1)
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(get_violations(heap), [])
        self.assertEqual(heap.head, None)
        self.assertEqual(heap.size, 0)

        heap = LeftistHeap(data)
        size = len(heap)
        while len(heap) > 1:
            curtop = heap.head
            top = heap.pop()
            size -= 1
            self.assertEqual(top, curtop)
            self.assertEqual(get_inversions(heap), [])
            self.assertEqual(get_violations(heap), [])
            self.assertEqual(heap.size, size)

    def test_replace(self):
        heap = LeftistHeap([1])
        top = heap.replace(0)
        self.assertEqual(top, 1)
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(get_violations(heap), [])
        self.assertEqual(heap.head, 0)
        self.assertEqual(heap.size, 1)

        heap = LeftistHeap([1, 2])
        top = heap.replace(0)
        self.assertEqual(top, 1)
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(get_violations(heap), [])
        self.assertEqual(heap.head, 0)
        self.assertEqual(heap.size, 2)

        heap = LeftistHeap(data)
        size = len(heap)
        for i in [1000] + list(range(-10, 10)) + [-1000]:
            curtop = heap.head
            top = heap.replace(i)
            self.assertEqual(top, curtop)
            self.assertEqual(get_inversions(heap), [])
            self.assertEqual(get_violations(heap), [])
            self.assertEqual(heap.size, size)

    def test_pushpop(self):
        heap = LeftistHeap([1])
        top = heap.pushpop(0)
        self.assertEqual(top, 0)
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(get_violations(heap), [])
        self.assertEqual(heap.head, 1)
        self.assertEqual(heap.size, 1)

        heap = LeftistHeap([1])
        top = heap.pushpop(2)
        self.assertEqual(top, 1)
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(get_violations(heap), [])
        self.assertEqual(heap.head, 2)
        self.assertEqual(heap.size, 1)

        heap = LeftistHeap()
        top = heap.pushpop(0)
        self.assertEqual(top, 0)
        self.assertEqual(get_inversions(heap), [])
        self.assertEqual(get_violations(heap), [])
        self.assertEqual(heap.head, None)
        self.assertEqual(heap.size, 0)

        heap = LeftistHeap(data)
        size = len(heap)
        for i in [1000] + list(range(-10, 10)) + [-1000]:
            target = min(heap.head, i)
            top = heap.pushpop(i)
            self.assertEqual(top, target)
            self.assertEqual(get_inversions(heap), [])
            self.assertEqual(get_violations(heap), [])
            self.assertEqual(heap.size, size)

    def test_find(self):
        heap = LeftistHeap([1])
        node = heap.find(1)
        self.assertEqual(node.value, 1)
        node = heap.find(2)
        self.assertEqual(node, None)
        node = heap.find(0)
        self.assertEqual(node, None)

        heap = LeftistHeap()
        node = heap.find(1)
        self.assertEqual(node, None)

        heap = LeftistHeap(data)
        for i in data:
            node = heap.find(i)
            self.assertEqual(node.value, i)
        self.assertEqual(heap.find(999), None)
        self.assertEqual(heap.find(-999), None)

    def test_update(self):
        heap = LeftistHeap()
        heap.update(0, 1)

        heap = LeftistHeap([0])
        heap.update(0, 1)
        self.assertEqual(heap.head, 1)

        heap = LeftistHeap(data)
        for i in data + [999, -999]:
            heap.update(i, choice(data + [999, -999]))
            self.assertEqual(get_inversions(heap), [])
            self.assertEqual(get_violations(heap), [])

    def test_remove(self):
        heap = LeftistHeap()
        heap.remove(0)
        self.assertEqual(heap.size, 0)
        self.assertEqual(heap.head, None)

        heap = LeftistHeap([0])
        heap.remove(1)
        self.assertEqual(heap.size, 1)
        self.assertEqual(heap.head, 0)
        heap.remove(0)
        self.assertEqual(heap.size, 0)
        self.assertEqual(heap.head, None)

        heap = LeftistHeap(data)
        for i in data + [100, 200, 300]:
            heap.remove(i)
            self.assertEqual(get_inversions(heap), [])
            self.assertEqual(get_violations(heap), [])


if __name__ == "__main__":
    unittest.main()
