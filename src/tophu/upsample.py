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
from typing import Iterable, Literal, SupportsIndex, SupportsInt, Tuple, Union, cast

import numpy as np
import scipy.fft
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "upsample",
]


IntOrInts = Union[SupportsInt, Iterable[SupportsInt]]
IndexOrIndicesOrNone = Union[SupportsIndex, Iterable[SupportsIndex], None]


def normalize_axis_tuple(axes: IndexOrIndicesOrNone, ndim: int) -> Tuple[int, ...]:
    """Normalize an axis argument into a tuple of nonnegative integer axes.

    Forbids any axis from being specified multiple times.

    Parameters
    ----------
    axes : int, iterable of int, or None
        Un-normalized axis or axes. `None` is treated as shorthand for all axes in an
        array of rank `ndim`.
    ndim : int
        The number of dimensions of the array that `axes` should be normalized against.

    Returns
    -------
    normalized_axes : tuple of int
        Normalized axis indices.
    """
    if axes is None:
        return tuple(range(ndim))
    else:
        return np.core.numeric.normalize_axis_tuple(axes, ndim)  # type: ignore


def fft_upsample(
    data: ArrayLike,
    ratio: IntOrInts,
    axes: IndexOrIndicesOrNone = None,
) -> NDArray:
    """Upsample using a Fast Fourier Transform (FFT)-based interpolation method.

    Upsample a discrete-time N-dimensional signal by zero-padding in the frequency
    domain. The input signal is assumed to be band-limited.

    Parameters
    ----------
    data : array_like
        Input array.
    ratio : int or iterable of int
        Upsampling ratio (along each axis to be upsampled). Must be greater than or
        equal to 1.
    axes : int, iterable of int, or None, optional
        Axis or axes to upsample. The default, `axes=None`, will upsample all axes in
        the input array.

    Returns
    -------
    out : numpy.ndarray
        Upsampled array.

    Notes
    -----
    If the length of `data` along any upsample axis is even-valued, the Discrete Fourier
    Transform (DFT) of `data` contains the Nyquist frequency bin, which can be
    ambiguously interpreted as a positive or negative frequency. In this case, the
    implementation follows the common approach of splitting the value of this cell in
    half among the positive and negative Nyquist bins when extending the frequency
    domain of the signal. This preserves Hermitian symmetry so that a resampled
    real-valued signal remains real-valued.
    """
    data = np.asanyarray(data)
    axes = normalize_axis_tuple(axes, data.ndim)

    # Normalize `ratio` into a tuple with the same length as `axes`. If `ratio` was a
    # scalar, upsample each axis by the same ratio.
    if isinstance(ratio, SupportsInt):
        r = int(ratio)
        ratio = (r,) * len(axes)
    else:
        ratio = tuple([int(r) for r in ratio])
        if len(ratio) != len(axes):
            raise ValueError(
                f"length mismatch: length of ratio ({len(ratio)}) must match number of"
                f" upsample axes ({len(axes)})"
            )

    # Convince static type checkers that `ratio` is a tuple of ints now.
    ratio = cast(Tuple[int, ...], ratio)

    # Check for invalid values of `ratio`.
    for r in ratio:
        if r < 1:
            raise ValueError("upsample ratio must be >= 1")

    # Check if input array is real-valued.
    real_input = np.isrealobj(data)

    # TODO: In case of real-valued input data, use `rfftn()`/`irfftn()` instead of
    # `fftn()`/`ifftn()` to speed up the FFT along the last axis (at the cost of
    # significantly more complicated implementation?)

    # Forward FFT.
    X = scipy.fft.fftn(data, axes=axes)

    # We will extend the frequency domain by zero-padding high frequencies. First, get
    # the shape of the zero-padded array.
    out_shape = list(data.shape)
    for r, axis in zip(ratio, axes):
        out_shape[axis] = data.shape[axis] * r

    # Initialize the padded array with zeros.
    Y = np.zeros(out_shape, dtype=X.dtype)

    # Get a slice object containing the positive frequency bins, including the DC and
    # Nyquist component (if present), of a frequency domain signal of length `n`.
    def posfreqbins(n: int) -> slice:
        return slice(None, n // 2 + 1)

    # Get a slice object containing the negative frequency bins, excluding the DC and
    # Nyquist component (if present), of a frequency domain signal of length `n`.
    def negfreqbins(n: int) -> slice:
        # If `n` is 2 or less, the sample frequencies include only the DC/Nyquist
        # frequency, so return an empty slice.
        if n <= 2:
            return slice(0, 0)
        return slice(-n // 2 + 1, None)

    # Copy the positive & negative frequency components of `X` to `Y` such that they are
    # separated by zero-padding.
    s = [slice(None)] * data.ndim
    for slice_funcs in itertools.product([posfreqbins, negfreqbins], repeat=len(axes)):
        for slice_func, axis in zip(slice_funcs, axes):
            n = data.shape[axis]
            s[axis] = slice_func(n)
        Y[tuple(s)] = X[tuple(s)]

    # Split the Nyquist frequency component in half among the positive & negative
    # Nyquist bin in the padded array.
    for axis in axes:
        n = data.shape[axis]
        if n % 2 == 0:
            s = [slice(None)] * data.ndim
            s[axis] = n // 2  # type: ignore
            Y[tuple(s)] *= 0.5
    for axis in axes:
        n = data.shape[axis]
        if n % 2 == 0:
            s1 = [slice(None)] * data.ndim
            s1[axis] = n // 2  # type: ignore
            s2 = [slice(None)] * data.ndim
            s2[axis] = -n // 2  # type: ignore
            Y[tuple(s2)] = Y[tuple(s1)]

    # Inverse FFT.
    y = scipy.fft.ifftn(Y, axes=axes, overwrite_x=True)

    # Rescale to compensate for the difference in forward/inverse FFT lengths.
    y *= np.prod(ratio)

    if real_input:
        return y.real
    else:
        return y


def upsample(
    data: ArrayLike,
    ratio: IntOrInts,
    axes: IndexOrIndicesOrNone = None,
    *,
    method: Literal["fft"] = "fft",
) -> NDArray:
    """Upsample an N-dimensional array.

    Parameters
    ----------
    data : array_like
        Input array.
    ratio : int or iterable of int
        Upsampling ratio (along each axis to be upsampled). Must be greater than or
        equal to 1.
    axes : int, iterable of int, or None, optional
        Axis or axes to upsample. The default, `axes=None`, will upsample all axes in
        the input array.
    method : {'fft'}, optional
        Upsampling method. Currently, only 'fft' is supported.

    Returns
    -------
    out : numpy.ndarray
        Upsampled array.
    """
    if method == "fft":
        return fft_upsample(data, ratio, axes)
    raise ValueError(f"unsupported method '{method}'")
