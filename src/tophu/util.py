from typing import Tuple, Union

import dask.array as da
import numpy as np
from dask.array.chunk import getitem
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "ceil_divide",
    "iseven",
    "map_blocks",
]


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


def iseven(n: int) -> bool:
    """Check if the input is even-valued."""
    return n % 2 == 0


def map_blocks(func, *args, **kwargs) -> Union[da.Array, Tuple[da.Array, ...]]:
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
