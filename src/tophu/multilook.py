# Copyright 2022 California Institute of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import warnings
from typing import Iterable, SupportsInt, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "multilook",
]


def multilook(arr: ArrayLike, nlooks: Union[int, Iterable[int]]) -> NDArray:
    """Multilook an array by simple averaging.

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
    if isinstance(nlooks, SupportsInt):
        n = int(nlooks)
        nlooks = (n,) * arr.ndim
    else:
        nlooks = tuple([int(n) for n in nlooks])
        if len(nlooks) != arr.ndim:
            raise ValueError(
                f"length mismatch: length of nlooks ({len(nlooks)}) must match input"
                f" array rank ({arr.ndim})"
            )

    # The number of looks must be at least 1 and at most the size of the input array
    # along the corresponding axis.
    for m, n in zip(arr.shape, nlooks):
        if n < 1:
            raise ValueError("number of looks must be >= 1")
        elif n > m:
            raise ValueError("number of looks should not exceed array shape")

    # Warn if the array shape is not an integer multiple of `nlooks`. Warn at most once
    # (even if multiple axes have this issue).
    for m, n in zip(arr.shape, nlooks):
        if m % n != 0:
            warnings.warn(
                "input array shape is not an integer multiple of nlooks -- remainder"
                " samples will be excluded from output",
                RuntimeWarning,
            )
            break

    # Initialize output array with zeros.
    out_shape = tuple([m // n for m, n in zip(arr.shape, nlooks)])
    out = np.zeros(out_shape, dtype=arr.dtype, like=arr)

    # Normalization factor (uniform weighting).
    w = 1.0 / np.prod(nlooks)

    # Compute the local average of samples by iteratively accumulating a weighted sum of
    # cells within each multilook window.
    subindices = (range(n) for n in nlooks)
    for index in itertools.product(*subindices):
        s = tuple(slice(i, j * k, k) for i, j, k in zip(index, out_shape, nlooks))
        out += w * arr[s]

    return out
