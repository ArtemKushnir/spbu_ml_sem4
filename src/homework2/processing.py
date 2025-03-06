from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Generic, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.homework2.kd_tree import T


class MetricCalculator(Generic[T]):
    def __init__(self, y_pred: NDArray[T], y_true: NDArray[T]) -> None:
        if y_pred.shape != y_true.shape:
            raise ValueError("'y_pred' and 'y_true' must have the same shape")
        self.y_pred: NDArray[T] = y_pred
        self.y_true: NDArray[T] = y_true
        self.len: int = y_true.size

    def true_positive(self) -> int:
        return int(np.sum((self.y_pred == 1) & (self.y_true == 1)))

    def false_positive(self) -> int:
        return int(np.sum((self.y_pred == 1) & (self.y_true != 1)))

    def true_negative(self) -> int:
        return int(np.sum((self.y_pred == 0) & (self.y_true == 0)))

    def false_negative(self) -> int:
        return int(np.sum((self.y_pred == 0) & (self.y_true != 0)))

    def calculate_recall(self) -> float:
        tp = self.true_positive()
        fn = self.false_negative()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def calculate_precision(self) -> float:
        tp = self.true_positive()
        fp = self.false_positive()
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def calculate_accuracy(self) -> float:
        return float(np.sum(self.y_pred == self.y_true) / self.len)

    def calculate_f1_score(self) -> float:
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


class Scaler(ABC):
    epsilon: float = 1e-8

    @abstractmethod
    def fit(self, data: NDArray[np.float64]) -> None:
        pass

    @abstractmethod
    def transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    def fit_transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        self.fit(data)
        return self.transform(data)

    def _is_fit(self) -> None:
        if not hasattr(self, "fitted") or not self.fitted:
            raise ValueError("Scaler is not fitted. Call 'fit()' first or call 'fit_transform()'")


class StandardScaler(Scaler):
    def __init__(self) -> None:
        self.std: Optional[NDArray[np.float64]] = None
        self.mean: Optional[NDArray[np.float64]] = None
        self.fitted: bool = False

    def fit(self, data: NDArray[np.float64]) -> None:
        self.std = np.std(data, axis=0) + self.epsilon
        self.mean = np.mean(data, axis=0)
        self.fitted = True

    def transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        self._is_fit()
        return (data - self.mean) / self.std  # type: ignore


class MinMaxScaler(Scaler):
    def __init__(self) -> None:
        self.min: Optional[NDArray[np.float64]] = None
        self.range: Optional[NDArray[np.float64]] = None
        self.fitted: bool = False

    def fit(self, data: NDArray[np.float64]) -> None:
        self.min = np.min(data, axis=0)
        self.range = np.ptp(data, axis=0) + self.epsilon
        self.fitted = True

    def transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        self._is_fit()
        return (data - self.min) / self.range  # type: ignore


class MaxAbsScaler(Scaler):
    def __init__(self) -> None:
        self.max_abs: Optional[NDArray[np.float64]] = None
        self.fitted: bool = False

    def fit(self, data: NDArray[np.float64]) -> None:
        self.max_abs = np.max(np.abs(data), axis=0) + self.epsilon
        self.fitted = True

    def transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        self._is_fit()
        return data / self.max_abs  # type: ignore


def train_test_split(
    X: NDArray[np.float64], y: NDArray[T], test_size: float, shuffle: bool = True, stratify: bool = False
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[T], NDArray[T]]:
    if not 0 <= test_size <= 1:
        raise ValueError("'test_size' must be between 0 and 1")
    if not shuffle and stratify:
        raise ValueError("Stratified train/test split is not implemented for shuffle=False")

    objects_cnt = len(y)
    train_cnt = int(objects_cnt * (1 - test_size))

    if not shuffle:
        return X[:train_cnt], X[train_cnt:], y[:train_cnt], y[train_cnt:]

    indexes = np.arange(X.shape[0])
    np.random.shuffle(indexes)

    if not stratify:
        return X[indexes[:train_cnt]], X[indexes[train_cnt:]], y[indexes[:train_cnt]], y[indexes[train_cnt:]]

    train_cnt_classes = {key: max(1, int(value * train_cnt / objects_cnt)) for key, value in Counter(y).items()}
    curr_cnt_classes: defaultdict[T, int] = defaultdict(int)

    train_mask = np.zeros(objects_cnt, dtype=bool)
    for i in range(objects_cnt):
        if curr_cnt_classes[y[i]] < train_cnt_classes[y[i]]:
            train_mask[i] = True
            curr_cnt_classes[y[i]] += 1

    return X[train_mask], X[~train_mask], y[train_mask], y[~train_mask]
