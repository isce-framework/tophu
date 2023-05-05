from typing import Tuple

import dask.array as da
import numpy as np
import pytest

import tophu


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
    shape: Tuple[int, ...] = (1024, 1024),
    chunks: Tuple[int, ...] = (128, 128),
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
