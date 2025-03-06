import heapq
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T", bound=np.generic)


@dataclass
class Point(Generic[T]):
    values: NDArray[np.float64]
    label: T


@dataclass
class Node:
    points: list[Point]
    split_dim: Optional[int] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None


@dataclass
class Neighbour:
    distance: float
    point: Point
    dim: int = field(init=False)

    def __post_init__(self) -> None:
        self.dim = len(self.point.values)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Neighbour):
            raise TypeError("The types of the element being compared must be 'Neighbor'")
        if self.dim != other.dim:
            raise ValueError("The dimensions don't match")
        return self.distance == other.distance

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Neighbour):
            raise TypeError("The types of the element being compared must be 'Neighbor'")
        if self.dim != other.dim:
            raise ValueError("The dimensions don't match")
        return self.distance < other.distance


class MaxHeap:
    def __init__(self, k: int) -> None:
        self.heap: list[Neighbour] = []
        self.size: int = 0
        self.k: int = k

    def push(self, neighbour: Neighbour) -> None:
        neighbour.distance *= -1
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, neighbour)
            self.size += 1
            return
        if neighbour.distance > self.heap[0].distance:
            heapq.heappushpop(self.heap, neighbour)

    def pop(self) -> Neighbour:
        neighbour = heapq.heappop(self.heap)
        self.size -= 1
        neighbour.distance *= -1
        return neighbour

    def get_k_min(self) -> list[Neighbour]:
        k_min_neighbour = [self.pop() for _ in range(self.k)][::-1]
        self.size = 0
        return k_min_neighbour

    def get_max_element(self) -> Optional[float]:
        return -self.heap[0].distance if len(self.heap) > 0 else None


class KDTree:
    _METRICS = {
        "euclidean": lambda a, b: float(np.linalg.norm(a - b)),
        "manhattan": lambda a, b: np.sum(np.abs(a - b)),
        "cosine": lambda a, b: 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)),
    }

    def __init__(self, X: list[Point], leaf_size: int = 1, metric: str = "euclidean") -> None:
        self.leaf_size: int = leaf_size
        self.dimension: int = len(X[0].values)
        self.size: int = len(X)
        self.head: Optional[Node] = self._create_tree(X)
        self.metric: str = metric

    def _create_tree(self, X: list[Point]) -> Optional[Node]:
        if len(X) == 0:
            return None
        if len(X) <= self.leaf_size:
            return Node(X)
        largest_spread_dim = self._get_largest_spread_dimension(X)
        X.sort(key=lambda x: x.values[largest_spread_dim])
        split_index = len(X) // 2
        curr_node = Node([X[split_index]], largest_spread_dim)
        curr_node.left = self._create_tree(X[:split_index])
        curr_node.right = self._create_tree((X[split_index + 1 :]))
        return curr_node

    @staticmethod
    def _get_largest_spread_dimension(X: list[Point]) -> int:
        data = np.array([point.values for point in X])
        return int(np.argmax(np.ptp(data, axis=0)))

    def query(self, X: NDArray[np.float64], k: int) -> list[list[Neighbour]]:
        k = min(k, self.size)
        result = []
        for row in X:
            max_heap = MaxHeap(k)
            self._get_k_near_neighbours(row, self.head, max_heap, k)
            result.append(max_heap.get_k_min())
        return result

    def _get_k_near_neighbours(
        self, point: NDArray[np.float64], curr_node: Optional[Node], max_heap: MaxHeap, k: int
    ) -> None:
        if curr_node is None:
            return
        if curr_node.left is None and curr_node.right is None:
            for curr_point in curr_node.points:
                max_heap.push(Neighbour(self._get_distance(point, curr_point.values), curr_point))
            return

        if curr_node.split_dim is None:
            raise TypeError("'split_dim' the dividing node cannot have None")

        if point[curr_node.split_dim] > curr_node.points[0].values[curr_node.split_dim]:
            self._get_k_near_neighbours(point, curr_node.right, max_heap, k)

            hyperplane_distance = abs(point[curr_node.split_dim] - curr_node.points[0].values[curr_node.split_dim])
            max_dist_neighbour = max_heap.get_max_element()

            if max_dist_neighbour is None or hyperplane_distance < max_dist_neighbour or max_heap.size < k:
                self._get_k_near_neighbours(point, curr_node.left, max_heap, k)
        else:
            self._get_k_near_neighbours(point, curr_node.left, max_heap, k)

            hyperplane_distance = abs(point[curr_node.split_dim] - curr_node.points[0].values[curr_node.split_dim])
            max_dist_neighbour = max_heap.get_max_element()

            if max_dist_neighbour is None or hyperplane_distance < max_dist_neighbour or max_heap.size < k:
                self._get_k_near_neighbours(point, curr_node.right, max_heap, k)
        max_heap.push(Neighbour(self._get_distance(point, curr_node.points[0].values), curr_node.points[0]))

    def _get_distance(self, first_point: NDArray[np.float64], second_point: NDArray[np.float64]) -> float:
        try:
            return self._METRICS[self.metric](first_point, second_point)
        except KeyError:
            raise ValueError(f"Unknown metric: {self.metric}")
