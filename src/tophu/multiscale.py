import warnings
from typing import Optional, Tuple

import numpy as np
import scipy.signal
from numpy.typing import ArrayLike, NDArray

from .filter import bandpass_equiripple_filter
from .multilook import multilook
from .tile import TiledPartition
from .unwrap import UnwrapCallback
from .upsample import upsample_nearest

__all__ = [
    "multiscale_unwrap",
]


def lowpass_filter_and_multilook(
    arr: ArrayLike,
    downsample_factor: Tuple[int, int],
    *,
    shape_factor: float = 1.5,
    overhang: float = 0.5,
    ripple: float = 0.01,
    attenuation: float = 40.0,
) -> NDArray:
    r"""
    Apply an anti-aliasing pre-filter, then multilook.

    The input array is filtered by applying a low-pass filter constructed using the
    optimal equiripple method\ :footcite:p:`mcclellan:1973` and then multilooked to
    produce the downsampled output. However, if the number of looks along any axis is 1,
    no filtering will be applied along that axis of the array.

    Parameters
    ----------
    arr : array_like
        The input data. A two-dimensional array.
    downsample_factor : tuple of int
        The number of looks to take along each axis of the input array.
    shape_factor : float, optional
        The shape factor of the filter (the ratio of the width of the combined
        pass-band and transition band to the pass-band width). Must be greater than or
        equal to 1. A larger shape factor results in a more gradual filter roll-off.
        (default: 1.5)
    overhang : float, optional
        The fraction of the low-pass filter transition bandwidth that extends beyond the
        Nyquist frequency of the resulting multilooked data. For example, if
        `overhang=0`, the transition band will be entirely within the Nyquist bandwidth.
        If `overhang=0.5`, the transition band will centered on the Nyquist frequency.
        The value must be within the interval [0, 1]. (default: 0.5)
    ripple : float, optional
        The maximum allowed ripple amplitude below unity gain in the pass-band, in
        decibels. (default: 0.01)
    attenuation : float, optional
        The stop-band attenuation (the difference in amplitude between the ideal gain in
        the pass-band and the highest gain in the stop-band), in decibels. (default: 40)

    Returns
    -------
    out : numpy.ndarray
        The output filtered and multilooked data.
    """
    arr = np.asanyarray(arr)

    if arr.ndim != 2:
        raise ValueError("input array must be 2-dimensional")
    if (overhang < 0.0) or (overhang > 1.0):
        raise ValueError("overhang must be between 0 and 1")
    if len(downsample_factor) != 2:
        raise ValueError("downsample factor should be a pair of ints")

    def get_filter_coeffs(n: int) -> NDArray:
        # If `n` is 1, then the subsequent multilooking step will be a no-op, so there's
        # no need to apply a low-pass filter. In that case, we choose the filter
        # coefficents to have a frequency response of unity.
        if n == 1:
            return 1.0

        # Get the ratio of the multilooked data sampling rate to the original data
        # sampling rate.
        ratio = 1.0 / n

        # Compute the normalized bandwidth of the pass-band (relative to the original
        # sampling rate), given the specified `shape_factor` and `overhang` values.
        bandwidth = ratio / (shape_factor * (1.0 - overhang) + overhang)

        # Get low pass filter coefficients. Force the filter length to be odd in order
        # to avoid applying a phase delay.
        return bandpass_equiripple_filter(
            bandwidth=bandwidth,
            shape=shape_factor,
            ripple=ripple,
            attenuation=attenuation,
            force_odd_length=True,
        )

    # Get low-pass filter coefficients for each axis to be multilooked. The combined 2-D
    # filter is the Cartesian product of the 1-D filters for each individual axis.
    coeffs1d = map(get_filter_coeffs, downsample_factor)
    coeffs2d = np.outer(*coeffs1d)

    # Filter the input data.
    filtered = scipy.signal.fftconvolve(arr, coeffs2d, mode="same")

    # Perform spatial averaging.
    return multilook(filtered, downsample_factor)


def upsample_unwrapped_phase(
    wrapped_phase_hires: NDArray[np.floating],
    wrapped_phase_lores: NDArray[np.floating],
    unwrapped_phase_lores: NDArray[np.floating],
    conncomp_lores: NDArray[np.unsignedinteger],
) -> NDArray[np.floating]:
    r"""
    Upsample an unwrapped phase field resulting from coarse unwrapping.

    Each pair of wrapped and unwrapped phase values are assumed to differ only by an
    integer number of cycles of :math:`2\pi`. In that case, we can form the upsampled
    unwrapped phase by upsampling the difference between the low-resolution unwrapped
    and wrapped phase as an integer number of cycles and then adding it to the
    high-resolution wrapped phase.

    Parameters
    ----------
    wrapped_phase_hires : numpy.ndarray
        The full-resolution wrapped phase, in radians. A two-dimensional array.
    wrapped_phase_lores : numpy.ndarray
        The low-resolution wrapped phase, in radians. A two-dimensional array.
    unwrapped_phase_lores : numpy.ndarray
        The unwrapped phase of the low-resolution interferogram, in radians. An array
        with the same shape as `wrapped_phase_lores`.
    conncomp_lores : numpy.ndarray
        Connected component labels associated with the low-resolution unwrapped phase.
        An array with the same shape as `igram_lores`. Each unique connected component
        should be assigned a positive integer label. Pixels not belonging to any
        connected component are considered invalid and should be labeled zero.

    Returns
    -------
    unwrapped_phase_hires : numpy.ndarray
        The upsampled unwrapped phase, in radians. An array with the same shape as
        `igram_hires`.

    Raises
    ------
    RuntimeError
        If the wrapped and unwrapped phase values are not congruent (if the observed
        difference between the wrapped and unwrapped phase values is not approximately
        an integer multiple of :math:`2\pi`).
    """
    # Estimate the number of cycles of phase difference between the unwrapped & wrapped
    # arrays.
    diff_cycles = (unwrapped_phase_lores - wrapped_phase_lores) / (2.0 * np.pi)
    diff_cycles_int = np.round(diff_cycles).astype(np.int32)

    # Get a mask of valid pixels (pixels belonging to any connected component).
    mask = conncomp_lores != 0

    # Check that the unwrapped & wrapped phase values are congruent (i.e. they differ
    # only by an integer multiple of 2pi) to within some absolute error tolerance.
    # Exclude invalid pixels since their phase values are not well-defined and may be
    # subject to implementation-specific behavior of different unwrapping algorithms.
    atol = 1.0e-3
    if not np.allclose(diff_cycles[mask], diff_cycles_int[mask], rtol=0.0, atol=atol):
        raise RuntimeError("wrapped and unwrapped phase values are not congruent")

    # Upsample the low-res offset between the unwrapped & wrapped phase.
    diff_cycles_hires = upsample_nearest(
        diff_cycles_int,
        out_shape=wrapped_phase_hires.shape,
    )

    # Get the upsampled coarse unwrapped phase field by adding multiples of 2pi to the
    # wrapped phase.
    return wrapped_phase_hires + 2.0 * np.pi * diff_cycles_hires


def coarse_unwrap(
    igram: NDArray[np.complexfloating],
    coherence: NDArray[np.floating],
    nlooks: float,
    unwrap: UnwrapCallback,
    downsample_factor: Tuple[int, int],
    *,
    do_lowpass_filter: bool = True,
    shape_factor: float = 1.5,
    overhang: float = 0.5,
    ripple: float = 0.01,
    attenuation: float = 40.0,
) -> Tuple[NDArray[np.floating], NDArray[np.unsignedinteger]]:
    """
    Estimate coarse unwrapped phase by unwrapping a downsampled interferogram.

    The input interferogram and coherence are multilooked down to lower resolution and
    then used to form a coarse unwrapped phase estimate, which is then upsampled back to
    the original sample spacing.

    A low-pass filter may be applied prior to multilooking the interferogram in order to
    reduce aliasing effects.

    Parameters
    ----------
    igram : numpy.ndarray
        The input interferogram. A two-dimensional complex-valued array.
    coherence : numpy.ndarray
        The sample coherence coefficient, with the same shape as the input
        interferogram.
    nlooks : float
        The effective number of looks used to form the input interferogram and
        coherence.
    unwrap : UnwrapCallback
        A callback function used to unwrap the interferogram at low resolution.
    downsample_factor : tuple of int
        The number of looks to take along each axis in order to form the low-resolution
        interferogram.
    do_lowpass_filter : bool, optional
        If True, apply a low-pass pre-filter prior to multilooking in order to reduce
        aliasing effects. (default: True)
    shape_factor : float, optional
        The shape factor of the filter (the ratio of the width of the combined
        pass-band and transition band to the pass-band width). Must be greater than or
        equal to 1. A larger shape factor results in a more gradual filter roll-off.
        Ignored if `do_lowpass_filter` is False. (default: 1.5)
    overhang : float, optional
        The fraction of the low-pass filter transition bandwidth that extends beyond the
        Nyquist frequency of the resulting multilooked data. For example, if
        `overhang=0`, the transition band will be entirely within the Nyquist bandwidth.
        If `overhang=0.5`, the transition band will centered on the Nyquist frequency.
        The value must be within the interval [0, 1]. Ignored if `do_lowpass_filter` is
        False. (default: 0.5)
    ripple : float, optional
        The maximum allowed ripple amplitude below unity gain in the pass-band, in
        decibels. Ignored if `do_lowpass_filter` is False. (default: 0.01)
    attenuation : float, optional
        The stop-band attenuation (the difference in amplitude between the ideal gain in
        the pass-band and the highest gain in the stop-band), in decibels. Ignored if
        `do_lowpass_filter` is False. (default: 40)

    Returns
    -------
    unwrapped_phase : numpy.ndarray
        The unwrapped phase, in radians. An array with the same shape as the input
        interferogram.
    conncomp : numpy.ndarray
        An array of connected component labels, with the same shape as the unwrapped
        phase.
    """
    # Get low-resolution interferogram & coherence data by multilooking. Optionally
    # low-pass filter the interferogram prior to multilooking to avoid aliasing.
    # Multilooking may raise warnings if the number of looks is even-valued or the input
    # array shape is not a multiple of the number of looks, both of which we can safely
    # ignore.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if do_lowpass_filter:
            igram_lores = lowpass_filter_and_multilook(
                igram,
                downsample_factor,
                shape_factor=shape_factor,
                overhang=overhang,
                ripple=ripple,
                attenuation=attenuation,
            )
        else:
            igram_lores = multilook(igram, downsample_factor)
        coherence_lores = multilook(coherence, downsample_factor)

    # Calculate the number of looks in the downsampled data.
    # XXX This isn't quite correct if the input data is single-look oversampled data
    # (`nlooks` should be 1 in that case, but `nlooks_lores` should ideally take into
    # account the spatial correlation of the signal that will reduce the effective
    # number of looks).
    nlooks_lores = nlooks * np.prod(downsample_factor)

    # Unwrap the downsampled data.
    unwrapped_phase, conncomp = unwrap(
        igram=igram_lores,
        corrcoef=coherence_lores,
        nlooks=nlooks_lores,
    )

    # Get the wrapped phase at each scale.
    wrapped_phase_hires = np.angle(igram)
    wrapped_phase_lores = np.angle(igram_lores)

    # Upsample unwrapped phase & connected component labels.
    unwrapped_phase_hires = upsample_unwrapped_phase(
        wrapped_phase_hires=wrapped_phase_hires,
        wrapped_phase_lores=wrapped_phase_lores,
        unwrapped_phase_lores=unwrapped_phase,
        conncomp_lores=conncomp,
    )
    conncomp_hires = upsample_nearest(conncomp, out_shape=igram.shape)

    return unwrapped_phase_hires, conncomp_hires


def adjust_conncomp_offset_cycles(
    unwrapped_phase_hires: NDArray[np.floating],
    conncomp_hires: NDArray[np.unsignedinteger],
    unwrapped_phase_lores: NDArray[np.floating],
    conncomp_lores: NDArray[np.unsignedinteger],
) -> None:
    r"""
    Remove phase cycle offsets from the high-resolution unwrapped phase.

    Attempt to correct for phase discontinuities in the unwrapped phase, such as those
    due to absolute phase ambiguities between independently unwrapped tiles, using a
    coarse estimate of the unwrapped phase as a reference. The high-resolution unwrapped
    phase is augmented by adding or subtracting cycles of :math:2\pi` to each connected
    component in order to minimize the mean discrepancy between the high-resolution and
    low-resolution phase values.

    The `unwrapped_phase_hires` array is modified in-place.

    Parameters
    ----------
    unwrapped_phase_hires : numpy.ndarray
        The high-resolution unwrapped phase, in radians.
    conncomp_hires : numpy.ndarray
        Connected component labels associated with the high-resolution unwrapped phase.
        An array with the same shape as `unwrapped_phase_hires`.
    unwrapped_phase_lores : numpy.ndarray
        A coarse estimate of the unwrapped phase, in radians. An array with the same
        shape as `unwrapped_phase_hires`.
    conncomp_lores : numpy.ndarray
        Connected component labels associated with the low-resolution unwrapped phase.
        An array with the same shape as `unwrapped phase_lores`.
    """
    # Get unique, non-zero connected component labels in the high-resolution data.
    unique_labels = set(np.unique(conncomp_hires))
    unique_nonzero_labels = unique_labels - {0}

    # For each connected component, determine the phase cycle offset between the
    # low-resolution and high-resolution unwrapped phase by computing the mean phase
    # difference, rounded to the nearest integer number of 2pi cycles.
    for label in unique_nonzero_labels:
        # Create a mask of pixels belonging to the current connected component in the
        # high-res data.
        conncomp_mask = conncomp_hires == label

        # Create a mask of pixels that are within the current connected component and
        # were valid (i.e. belonged to any connected component) in the low-res data.
        valid_mask = conncomp_mask & (conncomp_lores != 0)

        if np.any(valid_mask):
            # Compute the number of 2pi cycles that should be removed from the
            # high-resolution phase in order to minimize the mean difference between the
            # high-res and low-res phase.
            avg_offset = np.mean(
                unwrapped_phase_hires[valid_mask] - unwrapped_phase_lores[valid_mask]
            )
            avg_offset_cycles = np.round(avg_offset / (2.0 * np.pi))

            # Adjust the high-resolution phase in-place by subtracting a number of 2pi
            # phase cycles.
            unwrapped_phase_hires[conncomp_mask] -= 2.0 * np.pi * avg_offset_cycles


def multiscale_unwrap(
    igram: NDArray[np.complexfloating],
    coherence: NDArray[np.floating],
    nlooks: float,
    unwrap: UnwrapCallback,
    downsample_factor: Tuple[int, int],
    ntiles: Tuple[int, int],
    tile_overlap: Optional[Tuple[int, int]] = None,
    *,
    do_lowpass_filter: bool = True,
    shape_factor: float = 1.5,
    overhang: float = 0.5,
    ripple: float = 0.01,
    attenuation: float = 40.0,
) -> Tuple[NDArray[np.floating], NDArray[np.unsignedinteger]]:
    """
    Perform 2-D phase unwrapping using a multi-resolution approach.

    The input interferogram is broken up into smaller tiles, which are then unwrapped
    independently. In order to avoid phase discontinuities between tiles, additional
    phase cycles are added or subtracted within each tile in order to minimize the
    discrepancy with a coarse unwrapped phase estimate formed by multilooking the
    original interferogram.

    Parameters
    ----------
    igram : numpy.ndarray
        The input interferogram. A two-dimensional complex-valued array.
    coherence : numpy.ndarray
        The sample coherence coefficient, with the same shape as the input
        interferogram.
    nlooks : float
        The effective number of looks used to form the input interferogram and
        coherence.
    unwrap : UnwrapCallback
        A callback function used to unwrap the low-resolution interferogram and each
        high-resolution interferogram tile.
    downsample_factor : tuple of int
        The number of looks to take along each axis in order to form the low-resolution
        interferogram.
    ntiles : tuple of int
        The number of tiles along each axis. A pair of integers specifying the shape of
        the grid of tiles to partition the input interferogram into.
    tile_overlap : tuple of int or None, optional
        The overlap between adjacent tiles along each array axis, in samples.
        (default: None)
    do_lowpass_filter : bool, optional
        If True, apply a low-pass pre-filter prior to multilooking in order to reduce
        aliasing effects. (default: True)
    shape_factor : float, optional
        The shape factor of the anti-aliasing low-pass filter applied prior to
        multilooking (the ratio of the width of the combined pass-band and transition
        band to the pass-band width). A larger shape factor results in a more gradual
        filter roll-off. Ignored if `do_lowpass_filter` is False. (default: 1.5)
    overhang : float, optional
        The fraction of the low-pass filter transition bandwidth that extends beyond the
        Nyquist frequency of the resulting multilooked data. For example, if
        `overhang=0`, the transition band will be entirely within the Nyquist bandwidth.
        If `overhang=0.5`, the transition band will centered on the Nyquist frequency.
        The value must be within the interval [0, 1]. Ignored if `do_lowpass_filter` is
        False. (default: 0.5)
    ripple : float, optional
        The maximum allowed ripple amplitude below unity gain in the pass-band of the
        pre-filter, in decibels. Ignored if `do_lowpass_filter` is False.
        (default: 0.01)
    attenuation : float, optional
        The stop-band attenuation of the pre-filter (the difference in amplitude between
        the ideal gain in the pass-band and the highest gain in the stop-band), in
        decibels. Ignored if `do_lowpass_filter` is False. (default: 40)

    Returns
    -------
    unwrapped_phase : numpy.ndarray
        The unwrapped phase, in radians. An array with the same shape as the input
        interferogram.
    conncomp : numpy.ndarray
        An array of connected component labels, with the same shape as the unwrapped
        phase.
    """
    if igram.shape != coherence.shape:
        raise ValueError(
            "shape mismatch: interferogram and coherence arrays must have the same"
            " shape"
        )
    if nlooks < 1.0:
        raise ValueError("effective number of looks must be >= 1")
    if any(map(lambda d: d < 1, downsample_factor)):
        raise ValueError("downsample factor must be >= 1")
    if any(map(lambda n: n < 1, ntiles)):
        raise ValueError("number of tiles must be >= 1")

    # Check for the simple case where processing is single-tile and no additional
    # downsampling was requested. This case is functionally equivalent to just making a
    # single call to `unwrap()`.
    if ntiles == (1, 1) and downsample_factor == (1, 1):
        return unwrap(igram=igram, corrcoef=coherence, nlooks=nlooks)

    # Get a coarse estimate of the unwrapped phase using a low-resolution copy of the
    # interferogram.
    unwrapped_phase_lores, conncomp_lores = coarse_unwrap(
        igram=igram,
        coherence=coherence,
        nlooks=nlooks,
        unwrap=unwrap,
        downsample_factor=downsample_factor,
        do_lowpass_filter=do_lowpass_filter,
        shape_factor=shape_factor,
        overhang=overhang,
        ripple=ripple,
        attenuation=attenuation,
    )

    # Init output unwrapped phase and connected component labels.
    unwrapped_phase = np.zeros_like(igram, dtype=np.float32)
    conncomp = np.zeros_like(igram, dtype=np.uint32)

    # Partition the input interferogram into (possibly overlapping) tiles.
    tiles = TiledPartition(igram.shape, ntiles=ntiles, overlap=tile_overlap)

    for tile in tiles:
        # Unwrap each tile independently.
        unwrapped_phase[tile], conncomp[tile] = unwrap(
            igram=igram[tile],
            corrcoef=coherence[tile],
            nlooks=nlooks,
        )

        # Add or subtract multiples of 2pi to each connected component to minimize the
        # mean discrepancy between the high-res and low-res unwrapped phase (in order to
        # correct for phase discontinuities between adjacent tiles).
        adjust_conncomp_offset_cycles(
            unwrapped_phase[tile],
            conncomp[tile],
            unwrapped_phase_lores[tile],
            conncomp_lores[tile],
        )

    return unwrapped_phase, conncomp
