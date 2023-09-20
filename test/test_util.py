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


class TestGetAllUniqueValues:
    def test1(self):
        d1 = {"a": 0, "b": 1, "c": 2}
        d3 = {"d": 0, "e": 1, "f": 2}
        d2 = {"a": 3, "b": 4, "c": 5}
        d4 = {}
        unique_vals = tophu.get_all_unique_values([d1, d2, d3, d4])
        assert unique_vals == {0, 1, 2, 3, 4, 5}

    def test_empty(self):
        assert tophu.get_all_unique_values([]) == set()


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


class TestGetTileDims:
    @pytest.mark.parametrize(
        "shape,ntiles,tiledims",
        [
            ((99, 100), (3, 3), (33, 34)),
            ((60, 60, 60), (3, 4, 5), (20, 15, 12)),
            ((128, 129), (1, 1), (128, 129)),
            ((100, 100), (101, 1_000_000), (1, 1)),
        ],
    )
    def test_simple(
        self,
        shape: tuple[int, ...],
        ntiles: tuple[int, ...],
        tiledims: tuple[int, ...],
    ):
        assert tophu.get_tile_dims(shape, ntiles) == tiledims

    @pytest.mark.parametrize(
        "shape,ntiles,snap_to,tiledims",
        [
            ((30, 40, 50), (3, 4, 5), (5, 6, 7), (10, 12, 14)),
            ((99, 100), (3, 3), (5, 6), (35, 36)),
            ((60, 60, 60), (3, 4, 5), (3, 3, 3), (21, 15, 12)),
            ((128, 129), (1, 1), (200, 200), (128, 129)),
        ],
    )
    def test_snap_to(
        self,
        shape: tuple[int, ...],
        ntiles: tuple[int, ...],
        snap_to: tuple[int, ...],
        tiledims: tuple[int, ...],
    ):
        assert tophu.get_tile_dims(shape, ntiles, snap_to) == tiledims

    def test_size_mismatch(self):
        errmsg = r"^size mismatch: shape and ntiles must have same length$"
        with pytest.raises(ValueError, match=errmsg):
            tophu.get_tile_dims(shape=(100, 200, 300), ntiles=(1, 2))

        errmsg = r"^size mismatch: shape and snap_to must have same length$"
        with pytest.raises(ValueError, match=errmsg):
            tophu.get_tile_dims(shape=(100, 200, 300), ntiles=(1, 2, 3), snap_to=(1, 2))

    def test_bad_shape(self):
        with pytest.raises(ValueError, match=r"array axis lengths must be >= 1"):
            tophu.get_tile_dims(shape=(100, 0, 100), ntiles=(3, 3, 3))

    def test_bad_ntiles(self):
        with pytest.raises(ValueError, match=r"number of tiles must be >= 1"):
            tophu.get_tile_dims(shape=(100, 100, 100), ntiles=(3, 0, 3))

    def test_bad_snap_to(self):
        with pytest.raises(ValueError, match=r"snap_to lengths must be >= 1"):
            tophu.get_tile_dims(
                shape=(100, 100, 100),
                ntiles=(3, 3, 3),
                snap_to=(5, 0, 5),
            )


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


class TestMergeSets:
    def test1(self):
        assert tophu.merge_sets([{1, 2}, {3, 4}, {5, 6}]) == {1, 2, 3, 4, 5, 6}

    def test2(self):
        assert tophu.merge_sets([{1, 2}, {2, 3}, {3, 4}]) == {1, 2, 3, 4}

    def test_empty(self):
        assert tophu.merge_sets([]) == set()


class TestMode:
    def test_simple(self):
        arr = [0, 0, 1, 1, 1, 2, 2, 3]
        mode, count = tophu.mode(arr)
        assert mode == 1
        assert count == 3

    def test_multiple_modes(self):
        arr = [0, 1, 1, 1, 2, 2, 2, 3]
        mode, count = tophu.mode(arr)
        assert mode in (1, 2)
        assert count == 3

    def test_empty(self):
        mode, count = tophu.mode([])
        assert np.isnan(mode)
        assert count == 0

    def test_scalar(self):
        mode, count = tophu.mode(100.0)
        assert mode == 100.0
        assert count == 1


def test_round_up_to_next_multiple():
    assert tophu.round_up_to_next_multiple(5, 10) == 10
    assert tophu.round_up_to_next_multiple(10, 10) == 10
    assert tophu.round_up_to_next_multiple(11, 10) == 20
    assert tophu.round_up_to_next_multiple(-19, 10) == -10


def test_scratch_directory():
    with tophu.scratch_directory() as d1:
        assert d1.is_dir()
        with tophu.scratch_directory(d1) as d2:
            assert d2 == d1
            assert d2.is_dir()
        assert d2.is_dir()
    assert not d1.is_dir()


def test_unique_nonzero_integers():
    arr = [0, 0, 0, 1, 1, 2, 2, 2, 3]
    out = tophu.unique_nonzero_integers(arr)
    assert out == {1, 2, 3}
