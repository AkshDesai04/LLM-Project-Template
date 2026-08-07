"""
Unit tests for parallel executor utility (utils.concurrency.parallel_executor).
"""

import time
from unittest.mock import patch
import pytest

from utils.concurrency.parallel_executor import (
    ThreadSafeRateLimiter,
    _worker_wrapper,
    calculate_worker_count,
    parallel_execute,
)


@pytest.mark.concurrency
@pytest.mark.unit
def test_calculate_worker_count():
    with patch("os.cpu_count", return_value=4):
        assert calculate_worker_count(0, data_size=10) == 10
        assert calculate_worker_count(0, data_size=0) == 1
        assert calculate_worker_count(-1, data_size=10) == 4
        assert calculate_worker_count(-2, data_size=10) == 8
        assert calculate_worker_count(5, data_size=10) == 5


@pytest.mark.concurrency
@pytest.mark.unit
def test_rate_limiter():
    limiter = ThreadSafeRateLimiter(max_per_minute=600)  # interval = 0.1s
    start = time.time()
    limiter.wait_for_slot()
    limiter.wait_for_slot()
    elapsed = time.time() - start
    assert elapsed >= 0.05


@pytest.mark.concurrency
@pytest.mark.unit
def test_worker_wrapper_retries():
    mock_func = patch("utils.concurrency.parallel_executor.logger").start()
    calls = []

    def flaky_func(val):
        calls.append(val)
        if len(calls) < 2:
            raise ValueError("Temporary failure")
        return val * 2

    res = _worker_wrapper(flaky_func, (5,), max_retries=2, retry_timer=0.01)
    assert res == 10
    assert len(calls) == 2
    patch.stopall()


@pytest.mark.concurrency
@pytest.mark.unit
def test_worker_wrapper_max_retries_exceeded():
    def failing_func():
        raise RuntimeError("Permanent failure")

    res = _worker_wrapper(failing_func, (), max_retries=1, retry_timer=0.01)
    assert isinstance(res, RuntimeError)
    assert str(res) == "Permanent failure"


@pytest.mark.concurrency
@pytest.mark.unit
def test_parallel_execute_basic():
    def square(x):
        return x * x

    inputs = [1, 2, 3, 4, 5]
    results = parallel_execute(square, inputs, max_threads=2)
    assert results == [1, 4, 9, 16, 25]


@pytest.mark.concurrency
@pytest.mark.unit
def test_parallel_execute_tuple_unpacking():
    def add(a, b):
        return a + b

    inputs = [(1, 2), (3, 4), (5, 6)]
    results = parallel_execute(add, inputs, max_threads=2)
    assert results == [3, 7, 11]


@pytest.mark.concurrency
@pytest.mark.unit
def test_parallel_execute_empty():
    results = parallel_execute(lambda x: x, [], max_threads=1)
    assert results == []


@pytest.mark.concurrency
@pytest.mark.unit
def test_parallel_execute_order_and_failures():
    def mixed(val):
        if val == 3:
            raise ValueError("Error on 3")
        return val * 10

    inputs = [1, 2, 3, 4]
    results = parallel_execute(mixed, inputs, max_threads=2)
    assert results[0] == 10
    assert results[1] == 20
    assert isinstance(results[2], ValueError)
    assert results[3] == 40
