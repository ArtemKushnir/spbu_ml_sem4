from collections import Counter
from typing import Any, Optional

import numpy as np

from src.homework2.kd_tree import KDTree, Point


class KNNClassifier:
    def __init__(self, k: int, leaf_size: int, metric: str = "euclidean") -> None:
        self.k: int = k
        self.leaf_size: int = leaf_size
        self.metric: str = metric
        self.kd_tree: Optional[KDTree] = None
        self.classes: np.ndarray = np.array([])

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        if len(X_train) != len(y_train):
            raise ValueError("Number of training samples (X_train) does not match number of labels (y_train)")
        self.classes = np.unique(y_train)
        self.kd_tree = KDTree([Point(X_train[i], y_train[i]) for i in range(len(X_train))], self.leaf_size, self.metric)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        if self.kd_tree is None:
            raise ValueError("Model is not fitted. Call 'fit()' first")
        result = []
        k_near_neighbours = self.kd_tree.query(X_test, self.k)
        for point_neighbours in k_near_neighbours:
            labels_counter = Counter([neighbour.point.label for neighbour in point_neighbours])
            result.append([labels_counter[label] / self.k for label in self.classes])
        return np.array(result)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X_test)
        return np.array([self.classes[proba.index(max(proba))] for proba in probabilities])
