import itertools
import warnings
from typing import Iterable, SupportsInt, Tuple, Union, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "multilook",
]


IntOrInts = Union[SupportsInt, Iterable[SupportsInt]]


def iseven(n: int) -> bool:
    """Check if the input is even-valued."""
    return n % 2 == 0


def multilook(arr: ArrayLike, nlooks: IntOrInts) -> NDArray:
    """
    Multilook an array by simple averaging.

    Performs spatial averaging and decimation. Each element in the output array is the
    arithmetic mean of neighboring cells in the input array.

    Parameters
    ----------
    arr : array_like
        Input array.
    nlooks : int or iterable of int
        Number of looks along each axis of the input array.

    Returns
    -------
    out : numpy.ndarray
        Multilooked array.

    Notes
    -----
    If the length of the input array along a given axis is not evenly divisible by the
    specified number of looks, any remainder samples from the end of the array will be
    discarded in the output.
    """
    arr = np.asanyarray(arr)

    # Normalize `nlooks` into a tuple with length equal to `arr.ndim`. If `nlooks` was a
    # scalar, take the same number of looks along each axis in the array.
    try:
        n = int(nlooks)  # type: ignore
        nlooks = (n,) * arr.ndim
    except TypeError:
        nlooks = tuple([int(n) for n in nlooks])  # type: ignore
        if len(nlooks) != arr.ndim:
            raise ValueError(
                f"length mismatch: length of nlooks ({len(nlooks)}) must match input"
                f" array rank ({arr.ndim})"
            )

    # Convince static type checkers that `nlooks` is a tuple of ints now.
    nlooks = cast(Tuple[int, ...], nlooks)

    # The number of looks must be at least 1 and at most the size of the input array
    # along the corresponding axis.
    for m, n in zip(arr.shape, nlooks):
        if n < 1:
            raise ValueError("number of looks must be >= 1")
        elif n > m:
            raise ValueError("number of looks should not exceed array shape")

    # Warn if the number of looks along any axis is even-valued.
    if any(map(iseven, nlooks)):
        warnings.warn(
            (
                "one or more components of nlooks is even-valued -- this will result in"
                " a phase delay in the multilooked data equivalent to a half-bin shift"
            ),
            RuntimeWarning,
        )

    # Warn if any array dimensions are not integer multiples of `nlooks`.
    if any(m % n != 0 for (m, n) in zip(arr.shape, nlooks)):
        warnings.warn(
            (
                "input array shape is not an integer multiple of nlooks -- remainder"
                " samples will be excluded from output"
            ),
            RuntimeWarning,
        )

    # Initialize output array with zeros.
    out_shape = tuple([m // n for m, n in zip(arr.shape, nlooks)])
    out = np.zeros_like(arr, shape=out_shape)

    # Normalization factor (uniform weighting).
    w = 1.0 / np.prod(nlooks)

    # Now compute the local average of samples by iteratively accumulating a weighted
    # sum of cells within each multilook window.

    # Iterate over indices within the multilook kernel.
    subindices = (range(n) for n in nlooks)
    for index in itertools.product(*subindices):
        # Construct a strided multi-dimensional slice (a tuple of slice objects) that
        # accesses an element of the input array from each multilook window.
        start = index
        stop = np.multiply(nlooks, out_shape)
        step = nlooks
        ix = tuple([slice(a, b, c) for (a, b, c) in zip(start, stop, step)])

        # Accumulate the weighted sum of the input samples.
        out += w * arr[ix]

    return out
