from __future__ import annotations

from collections.abc import Iterable

import dask.array as da
import numpy as np
from dask.array.core import getitem
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "as_tuple_of_int",
    "ceil_divide",
    "get_lock",
    "iseven",
    "map_blocks",
    "round_up_to_next_multiple",
]


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
