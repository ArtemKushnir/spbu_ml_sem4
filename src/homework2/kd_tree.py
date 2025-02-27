import heapq
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Node:
    point_values: list[list[int]]
    split_dim: int = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None


@dataclass
class Neighbour:
    distance: int
    point: list[int]
    dim: int = field(init=False)

    def __post_init__(self):
        self.dim = len(self.point)

    def __eq__(self, other):
        if not isinstance(other, Neighbour):
            raise TypeError("The types of the element being compared must be 'Neighbor'")
        if self.dim != other.dim:
            raise ValueError("The dimensions don't match")
        return self.distance == other.distance

    def __lt__(self, other):
        if not isinstance(other, Neighbour):
            raise TypeError("The types of the element being compared must be 'Neighbor'")
        if self.dim != other.dim:
            raise ValueError("The dimensions don't match")
        return self.distance < other.distance


class MaxHeap:
    def __init__(self, k):
        self.heap = []
        self.size = 0
        self.k = k

    def push(self, neighbour: Neighbour):
        neighbour.distance *= -1
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, neighbour)
            self.size += 1
            return
        if neighbour.distance > self.heap[0].distance:
            heapq.heappushpop(self.heap, neighbour)

    def pop(self):
        neighbour = heapq.heappop(self.heap)
        self.size -= 1
        neighbour.distance *= -1
        return neighbour

    def get_k_min(self):
        k_min_neighbour = [self.pop() for _ in range(self.k)][::-1]
        self.size = 0
        return k_min_neighbour

    def get_max_element(self):
        return -self.heap[0].distance


class KDTree:
    def __init__(self, X, leaf_size=1):
        self.leaf_size = leaf_size
        self.dimension = len(X[0])
        self.size = len(X)
        self.head = self._create_tree(X)

    def _create_tree(self, X):
        if len(X) == 0:
            return None
        if len(X) <= self.leaf_size:
            return Node(X)
        largest_spread_dim = self._get_largest_spread_dimension(X)
        X.sort(key=lambda x: x[largest_spread_dim])
        split_index = len(X) // 2
        curr_node = Node([X[split_index]], largest_spread_dim)
        curr_node.left = self._create_tree(X[:split_index])
        curr_node.right = self._create_tree((X[split_index + 1 :]))
        return curr_node

    def _get_largest_spread_dimension(self, X):
        min_values = X[0].copy()
        max_values = X[0].copy()
        for row in X:
            min_values = [min(min_values[i], row[i]) for i in range(self.dimension)]
            max_values = [max(min_values[i], row[i]) for i in range(self.dimension)]
        result = [max_values[i] - min_values[i] for i in range(self.dimension)]
        return result.index(max(result))

    def query(self, X, k):
        result = []
        for row in X:
            if k > self.size:
                raise ValueError("k must be less than the size of the tree")
            max_heap = MaxHeap(k)
            self._get_k_near_neighbours(row, self.head, max_heap, k)
            result.append(max_heap.get_k_min())
        return result

    def _get_k_near_neighbours(self, point, curr_node: Node, max_heap: MaxHeap, k):
        if curr_node is None:
            return
        if curr_node.left is None and curr_node.right is None:
            for curr_point in curr_node.point_values:
                max_heap.push(Neighbour(self._get_len(point, curr_point), curr_point))
            return

        if point[curr_node.split_dim] > curr_node.point_values[0][curr_node.split_dim]:
            self._get_k_near_neighbours(point, curr_node.right, max_heap, k)

            hyperplane_distance = abs(point[curr_node.split_dim] - curr_node.point_values[0][curr_node.split_dim])
            max_dist_neighbour = max_heap.get_max_element()

            if hyperplane_distance < max_dist_neighbour or max_heap.size < k:
                self._get_k_near_neighbours(point, curr_node.left, max_heap, k)
        else:
            self._get_k_near_neighbours(point, curr_node.left, max_heap, k)

            hyperplane_distance = abs(point[curr_node.split_dim] - curr_node.point_values[0][curr_node.split_dim])
            max_dist_neighbour = max_heap.get_max_element()

            if hyperplane_distance < max_dist_neighbour or max_heap < k:
                self._get_k_near_neighbours(point, curr_node.right, max_heap, k)
        max_heap.push(Neighbour(self._get_len(point, curr_node.point_values[0]), curr_node.point_values[0]))

    def _get_len(self, first_point, second_point: list[int]):
        return sum([(first_point[i] - second_point[i]) ** 2 for i in range(self.dimension)]) ** 0.5
