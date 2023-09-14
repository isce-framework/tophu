from __future__ import annotations

import operator
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from functools import reduce
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypeVar

import dask.array as da
import numpy as np
from dask.array.core import getitem
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "as_tuple_of_int",
    "ceil_divide",
    "get_all_unique_values",
    "get_lock",
    "get_tile_dims",
    "iseven",
    "map_blocks",
    "merge_sets",
    "mode",
    "round_up_to_next_multiple",
    "scratch_directory",
    "unique_nonzero_integers",
]


# A generic type, without constraints.
T = TypeVar("T")


def as_tuple_of_int(ints: int | Iterable[int]) -> tuple[int, ...]:
    """
    Convert the input to a tuple of ints.

    Parameters
    ----------
    ints : int or iterable of int
        One or more integers.

    Returns
    -------
    out : tuple of int
        Tuple containing the inputs.
    """
    try:
        return (int(ints),)  # type: ignore
    except TypeError:
        return tuple([int(i) for i in ints])  # type: ignore


def ceil_divide(n: ArrayLike, d: ArrayLike) -> NDArray:
    """
    Return the smallest integer greater than or equal to the quotient of the inputs.

    Computes integer division of dividend `n` by divisor `d`, rounding up instead of
    truncating.

    Parameters
    ----------
    n : array_like
        Numerator.
    d : array_like
        Denominator.

    Returns
    -------
    q : numpy.ndarray
        Quotient.
    """
    n = np.asanyarray(n)
    d = np.asanyarray(d)
    return (n + d - np.sign(d)) // d


def get_all_unique_values(dicts: Iterable[dict[Any, T]]) -> set[T]:
    """Get the set of all unique values (not keys) in a sequence of dicts."""
    unique_values_per_dict = (set(d.values()) for d in dicts)
    return merge_sets(unique_values_per_dict)


def get_lock():
    """Get a lock appropriate for the current Dask scheduler."""
    import dask.base
    import dask.utils

    client_or_scheduler = dask.base.get_scheduler()

    try:
        import dask.distributed

        if isinstance(client_or_scheduler.__self__, dask.distributed.Client):
            return dask.distributed.Lock(client=client_or_scheduler)
    except (ImportError, AttributeError):
        pass

    return dask.utils.get_scheduler_lock(scheduler=client_or_scheduler)


def get_tile_dims(
    shape: tuple[int, ...],
    ntiles: tuple[int, ...],
    snap_to: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """
    Get tile dimensions of an array partitioned into tiles.

    Chooses tile dimensions such that an array of shape `shape` will be subdivided into
    blocks of roughly equal shape.

    Parameters
    ----------
    shape : tuple of int
        Shape of the array to be partitioned into tiles.
    ntiles : tuple of int
        Number of tiles along each array axis. Must be the same length as `shape`.
    snap_to : tuple of int or None, optional
        If specified, force tile dimensions to be a multiple of this value. Defaults to
        None.

    Returns
    -------
    tiledims : tuple of int
        Shape of a typical tile. The last tile along each axis may be smaller.
    """
    # Normalize `shape` and `ntiles` into tuples of ints.
    shape = as_tuple_of_int(shape)
    ntiles = as_tuple_of_int(ntiles)

    # Number of dimensions of the partitioned array.
    ndim = len(shape)

    if len(ntiles) != ndim:
        raise ValueError("size mismatch: shape and ntiles must have same length")
    if any(map(lambda s: s < 1, shape)):
        raise ValueError("array axis lengths must be >= 1")
    if any(map(lambda n: n < 1, ntiles)):
        raise ValueError("number of tiles must be >= 1")

    tiledims = ceil_divide(shape, ntiles)

    if snap_to is not None:
        # Normalize `snap_to` to a tuple of ints.
        snap_to = as_tuple_of_int(snap_to)

        if len(snap_to) != ndim:  # type: ignore[arg-type]
            raise ValueError("size mismatch: shape and snap_to must have same length")
        if any(map(lambda s: s < 1, snap_to)):  # type: ignore[arg-type]
            raise ValueError("snap_to lengths must be >= 1")

        tiledims = round_up_to_next_multiple(tiledims, snap_to)

    # Tile dimensions should not exceed the full array dimensions.
    tiledims = tuple(np.minimum(tiledims, shape))

    return tiledims


def iseven(n: int) -> bool:
    """Check if the input is even-valued."""
    return n % 2 == 0


def map_blocks(func, *args, **kwargs) -> da.Array | tuple[da.Array, ...]:
    """
    Map a function across all blocks of a dask array.

    Extends `dask.array.map_blocks` to handle functions with multiple return values. If
    `func` returns a tuple (or if the type of `meta`, if provided, is a tuple) then this
    returns a tuple of dask arrays instead of single array. Otherwise, the behavior is
    identical.

    See Also
    --------
    dask.array.map_blocks
    """
    out = da.map_blocks(func, *args, **kwargs)
    metas = out._meta
    if isinstance(metas, tuple):
        return tuple(
            da.map_blocks(getitem, out, index=i, meta=meta)
            for (i, meta) in enumerate(metas)
        )
    return out


def merge_sets(sets: Iterable[set[T]]) -> set[T]:
    """Return a new set that is the union of all of the input sets."""
    initial: set[T] = set()
    return reduce(operator.or_, sets, initial)


def mode(arr: ArrayLike) -> tuple[np.ndarray, int]:
    """
    Get the modal (most common) value in the input array.

    If there is more than one such value, only one is returned.

    The mode of an empty array is NaN, and the associated count is zero.

    Parameters
    ----------
    arr : array_like
        The input array.

    Returns
    -------
    mode : numpy.ndarray
        The modal value.
    count : int
        The number of times the mode appeared in the array.
    """
    arr = np.asanyarray(arr)

    if arr.size == 0:
        return np.nan, 0

    vals, counts = np.unique(arr, return_counts=True)
    idx = np.argmax(counts)
    return vals[idx], counts[idx]


def round_up_to_next_multiple(n: ArrayLike, base: ArrayLike) -> NDArray:
    """Round up to the next smallest multiple of `base`."""
    n = np.asanyarray(n)
    base = np.asanyarray(base)

    # Determine output datatype based on the inputs using NumPy's type promotion rules.
    # In particular, if both inputs were integer-valued, the result should also be
    # integer-valued.
    out_dtype = np.result_type(n, base)

    out = base * np.ceil(n / base)
    return np.require(out, dtype=out_dtype)


@contextmanager
def scratch_directory(d: str | os.PathLike | None = None, /) -> Iterator[Path]:
    """
    Context manager that creates a (possibly temporary) filesystem directory.

    If the input is a path-like object, a directory will be created at the
    specified filesystem path if it did not already exist. The directory will
    persist after leaving the context manager scope.

    If the input is None, a temporary directory is created as though by
    ``tempfile.TemporaryDirectory()``. Upon exiting the context manager scope, the
    directory and its contents are removed from the filesystem.

    Parameters
    ----------
    d : path-like or None, optional
        Scratch directory path. If None, a temporary directory is created. Defaults to
        None.

    Yields
    ------
    d : pathlib.Path
        Scratch directory path. If the input was None, the directory is removed
        from the filesystem upon exiting the context manager scope.
    """
    if d is None:
        try:
            tmpdir = TemporaryDirectory()
            yield Path(tmpdir.name)
        finally:
            tmpdir.cleanup()
    else:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        yield d


def unique_nonzero_integers(x: ArrayLike, /) -> set:
    """
    Find unique nonzero elements in the input array.

    Parameters
    ----------
    x : array_like
        An array of integers.

    Returns
    -------
    s : set
        The set of unique nonzero values.
    """
    return set(np.unique(x)) - {0}
