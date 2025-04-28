from collections import Counter
from typing import Generic, Optional

import numpy as np
from numpy.typing import NDArray

from src.homework2.kd_tree import KDTree, Point, T


class KNNClassifier(Generic[T]):
    def __init__(self, k: int, leaf_size: int, metric: str = "euclidean") -> None:
        self.k: int = k
        self.leaf_size: int = leaf_size
        self.metric: str = metric
        self.kd_tree: Optional[KDTree] = None
        self.classes: Optional[NDArray[T]] = None

    def fit(self, X_train: NDArray[np.float64], y_train: NDArray[T]) -> None:
        if len(X_train) != len(y_train):
            raise ValueError("Number of training samples (X_train) does not match number of labels (y_train)")
        self.classes = np.unique(y_train)
        self.kd_tree = KDTree([Point(X_train[i], y_train[i]) for i in range(len(X_train))], self.leaf_size, self.metric)

    def predict_proba(self, X_test: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.kd_tree is None or self.classes is None:
            raise ValueError("Model is not fitted. Call 'fit()' first")
        result = []
        k_near_neighbours = self.kd_tree.query(X_test, self.k)
        for point_neighbours in k_near_neighbours:
            labels_counter = Counter([neighbour.point.label for neighbour in point_neighbours])
            result.append([labels_counter[label] / self.k for label in self.classes])
        return np.array(result)

    def predict(self, X_test: NDArray[np.float64]) -> NDArray[T]:
        probabilities = self.predict_proba(X_test)
        return np.argmax(probabilities, axis=1)
