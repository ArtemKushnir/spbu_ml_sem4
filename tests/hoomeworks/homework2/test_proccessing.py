import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

from src.homework2.processing import MaxAbsScaler, MetricCalculator, MinMaxScaler, StandardScaler, train_test_split


@pytest.mark.parametrize(
    "y_pred, y_true, expected_accuracy",
    [
        (np.array([1, 0, 1, 1]), np.array([1, 0, 0, 1]), 0.75),
        (np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]), 1.0),
        (np.array([1, 1, 1, 1]), np.array([0, 0, 0, 0]), 0.0),
    ],
)
def test_metric_calculator_accuracy(y_pred, y_true, expected_accuracy):
    metric = MetricCalculator(y_pred, y_true)
    assert metric.calculate_accuracy() == expected_accuracy


@pytest.mark.parametrize("scaler_class", [StandardScaler, MinMaxScaler, MaxAbsScaler])
def test_scalers(scaler_class):
    data = np.random.rand(100, 5) * 100
    scaler = scaler_class()
    transformed = scaler.fit_transform(data)
    assert transformed.shape == data.shape

    if isinstance(scaler, StandardScaler):
        assert np.allclose(np.mean(transformed, axis=0), 0, atol=1e-6)
        assert np.allclose(np.std(transformed, axis=0), 1, atol=1e-6)
    elif isinstance(scaler, MinMaxScaler):
        assert np.allclose(np.min(transformed, axis=0), 0, atol=1e-6)
        assert np.allclose(np.max(transformed, axis=0), 1, atol=1e-6)
    elif isinstance(scaler, MaxAbsScaler):
        assert np.all(np.abs(transformed) <= 1)


@given(st.integers(min_value=50, max_value=200), st.floats(min_value=0.1, max_value=0.5))
def test_train_test_split(data_size, test_size):
    X = np.random.rand(data_size, 10)
    y = np.random.randint(0, 2, size=data_size)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    assert X_train.shape[0] + X_test.shape[0] == data_size
    assert y_train.shape[0] + y_test.shape[0] == data_size

    expected_test_size = int(data_size * test_size)
    assert abs(X_test.shape[0] - expected_test_size) <= 1
