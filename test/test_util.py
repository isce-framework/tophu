from __future__ import annotations

import multiprocessing
import threading
from multiprocessing.pool import ThreadPool

import dask
import dask.array as da
import numpy as np
import pytest
from dask.utils import SerializableLock

import tophu

# The type of object returned by `threading.Lock()` on the current platform.
ThreadingLock = type(threading.Lock())


def has_distributed() -> bool:
    """Check if the `dask.distributed` package is available."""
    try:
        import dask.distributed  # noqa: F401
    except ImportError:
        return False
    else:
        return True


class TestCeilDivide:
    def test_positive(self):
        assert tophu.ceil_divide(3, 2) == 2
        assert tophu.ceil_divide(4, 2) == 2
        assert tophu.ceil_divide(1, 1_000_000) == 1

    def test_negative(self):
        assert tophu.ceil_divide(-3, 2) == -1
        assert tophu.ceil_divide(3, -2) == -1
        assert tophu.ceil_divide(-3, -2) == 2

        assert tophu.ceil_divide(-4, 2) == -2
        assert tophu.ceil_divide(4, -2) == -2
        assert tophu.ceil_divide(-4, -2) == 2

    def test_zero(self):
        assert tophu.ceil_divide(0, 1) == 0
        assert tophu.ceil_divide(-0, 1) == 0

    def test_divide_by_zero(self):
        with pytest.warns(RuntimeWarning) as record:
            tophu.ceil_divide(1, 0)

        assert len(record) == 1

        message = record[0].message.args[0]
        assert "divide by zero encountered" in message

    def test_arraylike(self):
        result = tophu.ceil_divide([1, 2, 3, 4, 5], 2)
        expected = [1, 1, 2, 2, 3]
        np.testing.assert_array_equal(result, expected)


class TestGetLock:
    def test_no_scheduler(self):
        assert isinstance(tophu.get_lock(), (ThreadingLock, SerializableLock))

    def test_single_threaded(self):
        with dask.config.set(scheduler="synchronous"):
            assert isinstance(tophu.get_lock(), (ThreadingLock, SerializableLock))

    def test_threads_scheduler(self):
        with dask.config.set(scheduler="threads"):
            assert isinstance(tophu.get_lock(), (ThreadingLock, SerializableLock))

    def test_thread_pool(self):
        with dask.config.set(pool=ThreadPool(1)):
            assert isinstance(tophu.get_lock(), (ThreadingLock, SerializableLock))

    def test_processes_scheduler(self):
        with dask.config.set(scheduler="processes"):
            # XXX This returns a proxy object rather than an instance of
            # `multiprocessing.synchronize.Lock`.
            # Just test that the function succeeds for now -- not sure how to test this
            # robustly without getting too complicated.
            tophu.get_lock()

    def test_process_pool(self):
        with dask.config.set(pool=multiprocessing.Pool(1)):
            # XXX This returns a proxy object rather than an instance of
            # `multiprocessing.synchronize.Lock`.
            # Just test that the function succeeds for now -- not sure how to test this
            # robustly without getting too complicated.
            tophu.get_lock()

    @pytest.mark.skipif(
        not has_distributed(), reason="requires `dask.distributed` package"
    )
    def test_distributed(self):
        from dask.distributed import Client, Lock

        with Client():
            assert isinstance(tophu.get_lock(), Lock)


def test_iseven():
    assert tophu.iseven(2)
    assert tophu.iseven(-2)
    assert tophu.iseven(1 << 20)
    assert tophu.iseven(0)
    assert not tophu.iseven(1)
    assert not tophu.iseven(-5)
    assert not tophu.iseven((1 << 20) - 1)


def random_integer_array(
    low: int = 0,
    high: int = 100,
    shape: tuple[int, ...] = (1024, 1024),
    chunks: tuple[int, ...] = (128, 128),
) -> da.Array:
    return da.random.randint(low, high, size=shape, chunks=chunks)


class TestMapBlocks:
    def test_single_output(self):
        a = random_integer_array()
        b = random_integer_array()
        sum1 = da.map_blocks(np.add, a, b)
        sum2 = tophu.map_blocks(np.add, a, b)
        assert da.all(sum1 == sum2)

    def test_multiple_output(self):
        a = random_integer_array()
        b = random_integer_array(low=1)
        quot, rem = tophu.map_blocks(np.divmod, a, b)
        assert da.all(quot == a // b)
        assert da.all(rem == a % b)


def test_round_up_to_next_multiple():
    assert tophu.round_up_to_next_multiple(5, 10) == 10
    assert tophu.round_up_to_next_multiple(10, 10) == 10
    assert tophu.round_up_to_next_multiple(11, 10) == 20
    assert tophu.round_up_to_next_multiple(-19, 10) == -10
