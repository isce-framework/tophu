import itertools
from typing import Iterable, SupportsInt, Tuple, Union

import numpy as np
import scipy.fft
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "upsample_fft",
    "upsample_nearest",
]


IntOrInts = Union[SupportsInt, Iterable[SupportsInt]]


def as_tuple_of_int(ints: IntOrInts) -> Tuple[int, ...]:
    """
    Convert the input to a tuple of ints.

    Parameters
    ----------
    ints : int or iterable of int
        One or more integers.

    Returns
    -------
    out : tuple of int
        A tuple containing the inputs.
    """
    try:
        return (int(ints),)  # type: ignore
    except TypeError:
        return tuple([int(i) for i in ints])  # type: ignore


def validate_upsample_output_shape(
    in_shape: Tuple[int, ...],
    out_shape: Tuple[int, ...],
) -> None:
    """
    Check that `out_shape` is a valid upsampled output shape.

    The upsampled array should have the same number of dimensions as the input array,
    and its length along each axis should be greater than or equal to the original
    dimension.

    Parameters
    ----------
    in_shape : tuple of int
        The input array shape.
    out_shape : tuple of int
        The desired shape of the output array after upsampling.

    Raises
    ------
    ValueError
        If `out_shape` had the wrong number of dimensions or was smaller than the input
        array shape.
    """
    if len(out_shape) != len(in_shape):
        raise ValueError(
            f"length mismatch: length of out_shape ({len(out_shape)}) must match input"
            f" array rank ({len(in_shape)})"
        )
    if any(m < n for (m, n) in zip(out_shape, in_shape)):
        raise ValueError("output shape must be >= input data shape")


def iseven(n: int) -> bool:
    """Check if the input is even-valued."""
    return n % 2 == 0


def upsample_fft(data: ArrayLike, out_shape: IntOrInts) -> NDArray:
    """
    Upsample using a Fast Fourier Transform (FFT)-based interpolation method.

    Upsample a discrete-time N-dimensional signal by zero-padding in the frequency
    domain. The input signal is assumed to be band-limited.

    Parameters
    ----------
    data : array_like
        The input array.
    out_shape : int or iterable of int
        The desired shape of the output array after upsampling. Must be greater than or
        equal to the input array shape.

    Returns
    -------
    out : numpy.ndarray
        The upsampled array with shape `out_shape`.

    See Also
    --------
    upsample_nearest

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
    out_shape = as_tuple_of_int(out_shape)

    # Check that `out_shape` is valid.
    validate_upsample_output_shape(data.shape, out_shape)

    # Determine which axes to upsample.
    axes = [i for i in range(data.ndim) if (out_shape[i] != data.shape[i])]

    # TODO: In case of real-valued input data, use `rfftn()`/`irfftn()` instead of
    # `fftn()`/`ifftn()` to speed up the FFT along the last axis (at the cost of
    # significantly more complicated implementation?)

    # Forward FFT.
    X = scipy.fft.fftn(data, axes=axes)

    # We will extend the frequency domain by zero-padding high frequencies. First,
    # initialize the padded array with zeros.
    Y = np.zeros_like(X, shape=out_shape)

    # Get a slice object containing the positive frequency bins, including the DC and
    # Nyquist component (if present), of a frequency domain signal of length `n`.
    def posfreqbins(n: int) -> slice:
        # If `n` is 0, return an empty slice.
        if n == 0:
            return slice(0, 0)
        return slice(None, n // 2 + 1)

    # Get a slice object containing the negative frequency bins, excluding the DC and
    # Nyquist component (if present), of a frequency domain signal of length `n`.
    def negfreqbins(n: int) -> slice:
        # If `n` is 2 or less, the sample frequencies include only the DC and Nyquist
        # frequency (which NumPy/SciPy treats as positive), so return an empty slice.
        if n <= 2:
            return slice(0, 0)
        return slice(-n // 2 + 1, None)

    # Copy the positive & negative frequency components of `X` to `Y` such that they are
    # separated by zero-padding.
    for slice_funcs in itertools.product([posfreqbins, negfreqbins], repeat=len(axes)):
        s = [slice(None)] * data.ndim
        for slice_func, axis in zip(slice_funcs, axes):
            n = data.shape[axis]
            s[axis] = slice_func(n)
        Y[tuple(s)] = X[tuple(s)]

    # Split the Nyquist frequency component in half among the positive & negative
    # Nyquist bin in the padded array.
    for axis in axes:
        n = data.shape[axis]
        if iseven(n):
            s = [slice(None)] * data.ndim
            s[axis] = n // 2
            Y[tuple(s)] *= 0.5
    for axis in axes:
        n = data.shape[axis]
        if iseven(n):
            s1 = [slice(None)] * data.ndim
            s1[axis] = n // 2
            s2 = [slice(None)] * data.ndim
            s2[axis] = -n // 2
            Y[tuple(s2)] = Y[tuple(s1)]

    # Inverse FFT.
    y = scipy.fft.ifftn(Y, axes=axes, overwrite_x=True)

    # Rescale to compensate for the difference in forward/inverse FFT lengths.
    ratio = np.divide(out_shape, data.shape)
    y *= np.prod(ratio)

    # If input array was real-valued, the output should be as well.
    if np.isrealobj(data):
        return y.real
    else:
        return y


def pad_to_shape(
    arr: NDArray,
    out_shape: Tuple[int, ...],
    mode: str = "constant",
) -> NDArray:
    """
    Pad an array to the specified shape.

    Padding is added to the end of the array along each dimension where `out_shape` was
    greater than the input array shape.

    Parameters
    ----------
    arr : numpy.ndarray
        The input array.
    out_shape : tuple of int
        The shape of the output array after padding.
    mode : str or callable, optional
        The padding mode. See `numpy.pad()` for a list of possible modes.
        (default: 'constant')

    Returns
    -------
    out : numpy.ndarray
        The output padded array with shape equal to `out_shape`.
    """
    # Get the input array shape.
    in_shape = arr.shape

    # Check that the input array dimensions are all less than or equal to the output
    # padded dimensions.
    if not all(n <= m for (n, m) in zip(in_shape, out_shape)):
        raise ValueError("input array shape must not be larger than padded shape")

    # Determine how much padding to add to the end of each axis of the input array.
    padding = np.subtract(out_shape, in_shape)

    # Pad the array according to the specified `mode`.
    zeros = np.zeros_like(padding)
    return np.pad(arr, list(zip(zeros, padding)), mode=mode)


def upsample_nearest(data: ArrayLike, out_shape: IntOrInts) -> NDArray:
    """
    Upsample an array using nearest neighbor interpolation.

    The upsampled data is formed by duplicating elements in the input array. If any
    upsampled dimension is not an integer multiple of the input dimension, the last
    input sample along that axis will be repeated additional times to fill the remaining
    samples in the output.

    Parameters
    ----------
    data : array_like
        The input array.
    out_shape : int or iterable of int
        The desired shape of the output array after upsampling. Must be greater than or
        equal to the input array shape.

    Returns
    -------
    out : numpy.ndarray
        The upsampled array with shape `out_shape`.

    See Also
    --------
    upsample_fft
    """
    data = np.asanyarray(data)
    out_shape = as_tuple_of_int(out_shape)

    # Check that `out_shape` is valid.
    validate_upsample_output_shape(data.shape, out_shape)

    # Get the upsampling ratio, rounded down to the next largest integer if `out_shape`
    # was not an exact multiple of the input array shape.
    ratio = np.floor_divide(out_shape, data.shape)

    # Insert new dummy axes in both the input & output arrays of length 1 and `ratio`,
    # respectively, and let NumPy's broadcasting rules handle duplicating values.
    newaxes = np.arange(data.ndim) + 1
    s1 = np.insert(data.shape, newaxes, np.ones_like(ratio))
    s2 = np.insert(data.shape, newaxes, ratio)
    upsampled = np.empty_like(data, shape=s2)
    upsampled[:] = data.reshape(s1)

    # Reshape output array, collapsing dummy axes.
    newshape = ratio * data.shape
    upsampled = upsampled.reshape(newshape)

    # If `out_shape` was not an exact multiple of the input array shape, extend the
    # array to the specified output shape by padding with border values.
    if upsampled.shape == out_shape:
        return upsampled
    else:
        return pad_to_shape(upsampled, out_shape, mode="edge")
