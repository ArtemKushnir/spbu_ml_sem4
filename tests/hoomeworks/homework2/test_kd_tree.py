import math

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

from src.homework2.kd_tree import KDTree, Point


class TestKDTree:
    @staticmethod
    def find_k_near_neighbours(train_points, test_points, k):
        result = []
        for point in test_points:
            distance = []
            for train_point in train_points:
                distance.append(
                    (sum([(point[i] - train_point[i]) ** 2 for i in range(len(point))]) ** 0.5, train_point)
                )
            result.append(sorted(distance, key=lambda x: x[0])[:k])
        return result

    @given(
        st.integers(min_value=1, max_value=20),
        st.lists(
            st.lists(st.integers(min_value=-1000, max_value=1000), min_size=5, max_size=5), min_size=130, max_size=230
        ),
        st.integers(min_value=1, max_value=50),
    )
    def test_query(self, leaf_size, train_points, k):
        train_points = np.array(train_points)
        test_point = train_points[:30]
        train_points_without_label = train_points[30:]
        train_points_with_label = [Point(point, 1) for point in train_points_without_label]
        kd_tree = KDTree(train_points_with_label, leaf_size)
        k_near_neighbours = kd_tree.query(test_point, k)
        expected = self.find_k_near_neighbours(train_points_without_label, test_point, k)
        for i in range(30):
            for j in range(k):
                neighbour = k_near_neighbours[i][j]
                assert math.isclose(neighbour.distance, expected[i][j][0]) and np.allclose(
                    neighbour.point.values, expected[i][j][1]
                )
