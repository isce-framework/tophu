from typing import Protocol, Tuple, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "DatasetReader",
    "DatasetWriter",
]


@runtime_checkable
class DatasetReader(Protocol):
    """
    An array-like interface for reading input datasets.

    `DatasetReader` defines the abstract interface that types must conform to in order
    to be valid inputs to the `multiscale_unwrap()` function. Such objects must export
    NumPy-like `dtype`, `shape`, and `ndim` attributes and must support NumPy-style
    slice-based indexing.
    """

    dtype: np.dtype
    """numpy.dtype : Data-type of the array's elements."""

    shape: Tuple[int, ...]
    """tuple of int : Tuple of array dimensions."""

    ndim: int
    """int : Number of array dimensions."""

    def __getitem__(self, key: Tuple[slice, ...], /) -> ArrayLike:
        """Read a block of data."""
        ...


@runtime_checkable
class DatasetWriter(Protocol):
    """
    An array-like interface for writing output datasets.

    `DatasetWriter` defines the abstract interface that types must conform to in order
    to be valid outputs of the `multiscale_unwrap()` function. Such objects must export
    NumPy-like `dtype`, `shape`, and `ndim` attributes and must support NumPy-style
    slice-based indexing.
    """

    dtype: np.dtype
    """numpy.dtype : Data-type of the array's elements."""

    shape: Tuple[int, ...]
    """tuple of int : Tuple of array dimensions."""

    ndim: int
    """int : Number of array dimensions."""

    def __setitem__(self, key: Tuple[slice, ...], value: np.ndarray, /) -> None:
        """Write a block of data."""
        ...
