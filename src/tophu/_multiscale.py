from __future__ import annotations

import os
import warnings
from tempfile import NamedTemporaryFile

import dask.array as da
import numpy as np
import scipy.signal
from numpy.typing import ArrayLike, DTypeLike, NDArray

from . import _util
from ._filter import bandpass_equiripple_filter
from ._io import BinaryFile, DatasetReader, DatasetWriter
from ._label import relabel_hires_conncomps
from ._multilook import multilook
from ._unwrap import UnwrapCallback
from ._upsample import upsample_nearest

__all__ = [
    "multiscale_unwrap",
]


def lowpass_filter_and_multilook(
    arr: da.Array,
    downsample_factor: tuple[int, int],
    *,
    shape_factor: float = 1.5,
    overhang: float = 0.5,
    ripple: float = 0.01,
    attenuation: float = 40.0,
) -> da.Array:
    r"""
    Apply an anti-aliasing pre-filter, then multilook.

    The input array is filtered by applying a low-pass filter constructed using the
    optimal equiripple method\ :footcite:p:`mcclellan1973` and then multilooked to
    produce the downsampled output. However, if the number of looks along any axis is 1,
    no filtering will be applied along that axis of the array.

    Parameters
    ----------
    arr : dask.array.Array
        The input data. A two-dimensional array.
    downsample_factor : tuple of int
        The number of looks to take along each axis of the input array.
    shape_factor : float, optional
        The shape factor of the filter (the ratio of the width of the combined
        pass-band and transition band to the pass-band width). Must be greater than or
        equal to 1. A larger shape factor results in a more gradual filter roll-off.
        Defaults to 1.5.
    overhang : float, optional
        The fraction of the low-pass filter transition bandwidth that extends beyond the
        Nyquist frequency of the resulting multilooked data. For example, if
        `overhang=0`, the transition band will be entirely within the Nyquist bandwidth.
        If `overhang=0.5`, the transition band will centered on the Nyquist frequency.
        The value must be within the interval [0, 1]. Defaults to 0.5.
    ripple : float, optional
        The maximum allowed ripple amplitude below unity gain in the pass-band, in
        decibels. Defaults to 0.01.
    attenuation : float, optional
        The stop-band attenuation (the difference in amplitude between the ideal gain in
        the pass-band and the highest gain in the stop-band), in decibels. Defaults to
        40.

    Returns
    -------
    out : dask.array.Array
        The output filtered and multilooked data.
    """
    if arr.ndim != 2:
        raise ValueError("input array must be 2-dimensional")
    if (overhang < 0.0) or (overhang > 1.0):
        raise ValueError("overhang must be between 0 and 1")
    if len(downsample_factor) != 2:
        raise ValueError("downsample factor should be a pair of ints")

    def get_filter_coeffs(n: int) -> NDArray:
        # If `n` is 1, then the subsequent multilooking step will be a no-op, so there's
        # no need to apply a low-pass filter. In that case, we choose the filter
        # coefficients to have a frequency response of unity.
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
    depth = tuple(np.floor_divide(coeffs2d.shape, 2))
    filtered = da.map_overlap(
        scipy.signal.fftconvolve,
        arr,
        depth=depth,
        boundary=0.0,
        trim=True,
        in2=coeffs2d,
        mode="same",
    )

    # Perform spatial averaging.
    return multilook(filtered, downsample_factor)


def upsample_unwrapped_phase(
    wrapped_hires: da.Array,
    wrapped_lores: da.Array,
    unwrapped_lores: da.Array,
    conncomp_lores: da.Array,
) -> da.Array:
    r"""
    Upsample an unwrapped phase field resulting from coarse unwrapping.

    Each pair of wrapped and unwrapped phase values are assumed to differ only by an
    integer number of cycles of :math:`2\pi`. In that case, we can form the upsampled
    unwrapped phase by upsampling the difference between the low-resolution unwrapped
    and wrapped phase as an integer number of cycles and then adding it to the
    high-resolution wrapped phase.

    Parameters
    ----------
    wrapped_hires : dask.array.Array
        The full-resolution wrapped phase, in radians. A two-dimensional array.
    wrapped_lores : dask.array.Array
        The low-resolution wrapped phase, in radians. A two-dimensional array.
    unwrapped_lores : dask.array.Array
        The unwrapped phase of the low-resolution interferogram, in radians. An array
        with the same shape as `wrapped_lores`.
    conncomp_lores : dask.array.Array
        Connected component labels associated with the low-resolution unwrapped phase.
        An array with the same shape as `igram_lores`. Each unique connected component
        should be assigned a positive integer label. Pixels not belonging to any
        connected component are considered invalid and should be labeled zero.

    Returns
    -------
    unwrapped_hires : dask.array.Array
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
    diff_cycles = (unwrapped_lores - wrapped_lores) / (2.0 * np.pi)
    diff_cycles_int = da.round(diff_cycles).astype(np.int32)

    def check_congruence(
        diff_cycles: np.ndarray,
        diff_cycles_int: np.ndarray,
        conncomp: np.ndarray,
        atol: float = 1e-3,
    ) -> None:
        mask = conncomp != 0
        if not np.allclose(
            diff_cycles[mask], diff_cycles_int[mask], rtol=0.0, atol=atol
        ):
            raise RuntimeError("wrapped and unwrapped phase values are not congruent")

    # Check that the unwrapped & wrapped phase values are congruent (i.e. they differ
    # only by an integer multiple of 2pi) to within some absolute error tolerance.
    # Exclude invalid pixels since their phase values are not well-defined and may be
    # subject to implementation-specific behavior of different unwrapping algorithms.
    da.map_blocks(check_congruence, diff_cycles, diff_cycles_int, conncomp_lores)

    # Upsample the low-res offset between the unwrapped & wrapped phase.
    diff_cycles_hires = upsample_nearest(
        diff_cycles_int,
        out_shape=wrapped_hires.shape,
    )

    # Ensure that diff_cycles and wrapped phase have the same chunksizes after
    # upsampling.
    if diff_cycles_hires.chunks != wrapped_hires.chunks:
        diff_cycles_hires = diff_cycles_hires.rechunk(wrapped_hires.chunks)

    # Get the upsampled coarse unwrapped phase field by adding multiples of 2pi to the
    # wrapped phase.
    return wrapped_hires + 2.0 * np.pi * diff_cycles_hires


def coarse_unwrap(
    igram: da.Array,
    coherence: da.Array,
    nlooks: float,
    unwrap_func: UnwrapCallback,
    downsample_factor: tuple[int, int],
    *,
    do_lowpass_filter: bool = True,
    shape_factor: float = 1.5,
    overhang: float = 0.5,
    ripple: float = 0.01,
    attenuation: float = 40.0,
) -> tuple[da.Array, da.Array]:
    """
    Estimate coarse unwrapped phase by unwrapping a downsampled interferogram.

    The input interferogram and coherence are multilooked down to lower resolution and
    then used to form a coarse unwrapped phase estimate, which is then upsampled back to
    the original sample spacing.

    A low-pass filter may be applied prior to multilooking the interferogram in order to
    reduce aliasing effects.

    Parameters
    ----------
    igram : dask.array.Array
        The input interferogram. A two-dimensional complex-valued array.
    coherence : dask.array.Array
        The sample coherence coefficient, with the same shape as the input
        interferogram.
    nlooks : float
        The effective number of looks used to form the input interferogram and
        coherence.
    unwrap_func : UnwrapCallback
        A callback function used to unwrap the interferogram at low resolution.
    downsample_factor : tuple of int
        The number of looks to take along each axis in order to form the low-resolution
        interferogram.
    do_lowpass_filter : bool, optional
        If True, apply a low-pass pre-filter prior to multilooking in order to reduce
        aliasing effects. Defaults to True.
    shape_factor : float, optional
        The shape factor of the filter (the ratio of the width of the combined
        pass-band and transition band to the pass-band width). Must be greater than or
        equal to 1. A larger shape factor results in a more gradual filter roll-off.
        Ignored if `do_lowpass_filter` is False. Defaults to 1.5.
    overhang : float, optional
        The fraction of the low-pass filter transition bandwidth that extends beyond the
        Nyquist frequency of the resulting multilooked data. For example, if
        `overhang=0`, the transition band will be entirely within the Nyquist bandwidth.
        If `overhang=0.5`, the transition band will centered on the Nyquist frequency.
        The value must be within the interval [0, 1]. Ignored if `do_lowpass_filter` is
        False. Defaults to 0.5.
    ripple : float, optional
        The maximum allowed ripple amplitude below unity gain in the pass-band, in
        decibels. Ignored if `do_lowpass_filter` is False. Defaults to 0.01.
    attenuation : float, optional
        The stop-band attenuation (the difference in amplitude between the ideal gain in
        the pass-band and the highest gain in the stop-band), in decibels. Ignored if
        `do_lowpass_filter` is False. Defaults to 40.

    Returns
    -------
    unwrapped : dask.array.Array
        The unwrapped phase, in radians. An array with the same shape as the input
        interferogram.
    conncomp : dask.array.Array
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

    # XXX This is a hack to try and trick Dask into doing something that it should be
    # smart enough to do on its own, but inexplicably does not.
    #
    # We're converting the input Dask array into a NumPy array and then back into a Dask
    # array with a single chunk.
    #
    # All we really want to do here is rechunk the input array to a single chunk so that
    # we can unwrap the full array as a single block. (Even though we're just unwrapping
    # a single block, we still want to perform this operation using Dask so that it can
    # run in parallel with other unwrapping steps elsewhere in the code.)
    #
    # For some unknown reason, though, if we simply rechunk the input in the usual way,
    # Dask will wait until one or more other unwrapping steps are finished before this
    # one starts (even though they have no interdependencies and this one is actually
    # queued in the task graph first!). It seems to do this regardless of how many
    # parallel workers we give it.
    #
    # Doing the rechunking this way seems to convince Dask that it can, in fact, perform
    # this unwrapping step independently of all the others, with potentially huge
    # benefits to overall runtime. So it stays until I can figure what's going on.
    def to_single_chunk(arr: ArrayLike) -> da.Array:
        return da.from_array(np.asarray(arr), chunks=arr.shape)

    igram_lores_singleblock = to_single_chunk(igram_lores)
    coherence_lores_singleblock = to_single_chunk(coherence_lores)

    # Unwrap the downsampled data.
    unwrapped_lores, conncomp_lores = _util.map_blocks(
        unwrap_func,
        igram_lores_singleblock,
        coherence_lores_singleblock,
        nlooks=nlooks_lores,
        meta=(np.empty((), dtype=np.float32), np.empty((), dtype=np.uint32)),
    )

    unwrapped_lores = unwrapped_lores.rechunk(igram_lores.chunks)
    conncomp_lores = conncomp_lores.rechunk(igram_lores.chunks)

    # Get the wrapped phase at each scale.
    wrapped_hires = da.angle(igram)
    wrapped_lores = da.angle(igram_lores)

    # Upsample unwrapped phase & connected component labels.
    unwrapped_hires = upsample_unwrapped_phase(
        wrapped_hires=wrapped_hires,
        wrapped_lores=wrapped_lores,
        unwrapped_lores=unwrapped_lores,
        conncomp_lores=conncomp_lores,
    )
    conncomp_hires = upsample_nearest(conncomp_lores, out_shape=igram.shape)

    # Ensure that connected components and unwrapped phase have the same chunksizes
    # after upsampling.
    if conncomp_hires.chunks != unwrapped_hires.chunks:
        conncomp_hires = conncomp_hires.rechunk(unwrapped_hires.chunks)

    return unwrapped_hires, conncomp_hires


def adjust_conncomp_offset_cycles(
    unwrapped_hires: NDArray[np.floating],
    conncomp_hires: NDArray[np.unsignedinteger],
    unwrapped_lores: NDArray[np.floating],
    conncomp_lores: NDArray[np.unsignedinteger],
) -> NDArray[np.floating]:
    r"""
    Remove phase cycle offsets from the high-resolution unwrapped phase.

    Attempt to correct for phase discontinuities in the unwrapped phase, such as those
    due to absolute phase ambiguities between independently unwrapped tiles, using a
    coarse estimate of the unwrapped phase as a reference. The high-resolution unwrapped
    phase is augmented by adding or subtracting cycles of :math:2\pi` to each connected
    component in order to minimize the mean discrepancy between the high-resolution and
    low-resolution phase values.

    Parameters
    ----------
    unwrapped_hires : numpy.ndarray
        The high-resolution unwrapped phase, in radians.
    conncomp_hires : numpy.ndarray
        Connected component labels associated with the high-resolution unwrapped phase.
        An array with the same shape as `unwrapped_hires`.
    unwrapped_lores : numpy.ndarray
        A coarse estimate of the unwrapped phase, in radians. An array with the same
        shape as `unwrapped_hires`.
    conncomp_lores : numpy.ndarray
        Connected component labels associated with the low-resolution unwrapped phase.
        An array with the same shape as `unwrapped_lores`.

    Returns
    -------
    new_unwrapped_hires : numpy.ndarray
        The corrected high-resolution unwrapped phase, in radians.
    """
    # Get unique, non-zero connected component labels in the high-resolution data.
    unique_labels = _util.unique_nonzero_integers(conncomp_hires)

    new_unwrapped_hires = np.copy(unwrapped_hires)

    # For each connected component, determine the phase cycle offset between the
    # low-resolution and high-resolution unwrapped phase by computing the mean phase
    # difference, rounded to the nearest integer number of 2pi cycles.
    for label in unique_labels:
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
                unwrapped_hires[valid_mask] - unwrapped_lores[valid_mask]
            )
            avg_offset_cycles = np.round(avg_offset / (2.0 * np.pi))

            # Adjust the output unwrapped phase by subtracting a number of 2pi phase
            # cycles.
            new_unwrapped_hires[conncomp_mask] -= 2.0 * np.pi * avg_offset_cycles

    return new_unwrapped_hires


def unique_binary_file(
    dir: str | os.PathLike,
    shape: tuple[int, ...],
    dtype: DTypeLike,
    prefix: str | None = None,
    suffix: str | None = None,
) -> BinaryFile:
    """
    Create a new `BinaryFile` with a unique file name.

    Parameters
    ----------
    dir : path-like
        The file system path of the directory to create the file within. Must be an
        existing directory.
    shape : tuple of int
        Tuple of array dimensions.
    dtype : data-type
        Data-type of the array's elements. Must be convertible to a `numpy.dtype`
        object.
    prefix : str or None, optional
        If not None, the file name will begin with this prefix. Otherwise, a default
        prefix is used. Defaults to None.
    suffix : str or None, optional
        If not None, the file name will end with this suffix. Otherwise, there will be
        no suffix. Defaults to None.

    Returns
    -------
    binary_file : BinaryFile
        A raw binary file in the specified directory with a unique file name.
    """
    dir = os.fspath(dir)
    file = NamedTemporaryFile(suffix=suffix, prefix=prefix, dir=dir, delete=False)
    return BinaryFile(filepath=file.name, shape=shape, dtype=dtype)


def multiscale_unwrap(
    unwrapped: DatasetWriter,
    conncomp: DatasetWriter,
    igram: DatasetReader,
    coherence: DatasetReader,
    nlooks: float,
    unwrap_func: UnwrapCallback,
    downsample_factor: tuple[int, int],
    ntiles: tuple[int, int],
    min_conncomp_overlap: float = 0.5,
    scratchdir: str | os.PathLike | None = None,
    *,
    do_lowpass_filter: bool = True,
    shape_factor: float = 1.5,
    overhang: float = 0.5,
    ripple: float = 0.01,
    attenuation: float = 40.0,
) -> None:
    """
    Perform 2-D phase unwrapping using a multi-resolution approach.

    The input interferogram is broken up into smaller tiles, which are then unwrapped
    independently. In order to avoid phase discontinuities between tiles, additional
    phase cycles are added or subtracted within each tile in order to minimize the
    discrepancy with a coarse unwrapped phase estimate formed by multilooking the
    original interferogram.

    Parameters
    ----------
    unwrapped : DatasetWriter
        The output unwrapped phase, in radians. An array with the same shape as the
        input interferogram.
    conncomp : DatasetWriter
        The output array of connected component labels, with the same shape as the
        unwrapped phase.
    igram : DatasetReader
        The input interferogram. A two-dimensional complex-valued array. If `igram` has
        a ``.chunks`` attribute, then the chosen tile dimensions will be a multiple of
        that chunk shape.
    coherence : DatasetReader
        The sample coherence coefficient, with the same shape as the input
        interferogram.
    nlooks : float
        The effective number of looks used to form the input interferogram and
        coherence.
    unwrap_func : UnwrapCallback
        A callback function used to unwrap the low-resolution interferogram and each
        high-resolution interferogram tile.
    downsample_factor : tuple of int
        The number of looks to take along each axis in order to form the low-resolution
        interferogram.
    ntiles : tuple of int
        The number of tiles along each axis. A pair of integers specifying the shape of
        the grid of tiles to partition the input interferogram into.
    min_conncomp_overlap : float, optional
        Minimum intersection between components in order to be considered overlapping,
        as a fraction of the area of the high-resolution component area. Used during the
        relabeling step to determine overlap between connected components resulting from
        tiled unwrapping and those resulting from coarse unwrapping. Must be in the
        range (0, 1]. Defaults to 0.5.
    scratchdir : path-like or None, optional
        Scratch directory where intermediate processing artifacts are written.
        If the specified directory does not exist, it will be created. If None,
        a temporary directory will be created and automatically removed from the
        filesystem at the end of processing. Otherwise, the directory and its
        contents will not be cleaned up. Defaults to None.
    do_lowpass_filter : bool, optional
        If True, apply a low-pass pre-filter prior to multilooking in order to reduce
        aliasing effects. Defaults to True.
    shape_factor : float, optional
        The shape factor of the anti-aliasing low-pass filter applied prior to
        multilooking (the ratio of the width of the combined pass-band and transition
        band to the pass-band width). A larger shape factor results in a more gradual
        filter roll-off. Ignored if `do_lowpass_filter` is False. Defaults to 1.5.
    overhang : float, optional
        The fraction of the low-pass filter transition bandwidth that extends beyond the
        Nyquist frequency of the resulting multilooked data. For example, if
        `overhang=0`, the transition band will be entirely within the Nyquist bandwidth.
        If `overhang=0.5`, the transition band will centered on the Nyquist frequency.
        The value must be within the interval [0, 1]. Ignored if `do_lowpass_filter` is
        False. Defaults to 0.5.
    ripple : float, optional
        The maximum allowed ripple amplitude below unity gain in the pass-band of the
        pre-filter, in decibels. Ignored if `do_lowpass_filter` is False.
        Defaults to 0.01.
    attenuation : float, optional
        The stop-band attenuation of the pre-filter (the difference in amplitude between
        the ideal gain in the pass-band and the highest gain in the stop-band), in
        decibels. Ignored if `do_lowpass_filter` is False. Defaults to 40.
    """
    if unwrapped.shape != igram.shape:
        raise ValueError("shape mismatch: igram and unwrapped must have the same shape")
    if conncomp.shape != unwrapped.shape:
        raise ValueError(
            "shape mismatch: unwrapped and conncomp must have the same shape"
        )
    if igram.shape != coherence.shape:
        raise ValueError("shape mismatch: igram and coherence must have the same shape")

    if nlooks < 1.0:
        raise ValueError("effective number of looks must be >= 1")

    if any(map(lambda d: d < 1, downsample_factor)):
        raise ValueError("downsample factor must be >= 1")

    # Get chunksize. If the input has a `chunks` attribute (e.g. h5py Datasets, zarr
    # Arrays), ensure that the chunksize is a multiple of that shape.
    if hasattr(igram, "chunks"):
        chunksize = _util.get_tile_dims(igram.shape, ntiles, snap_to=igram.chunks)
    else:
        chunksize = _util.get_tile_dims(igram.shape, ntiles)

    # Convert inputs to dask arrays. Interferogram and coherence must have the same
    # chunk sizes.
    igram_ = da.from_array(igram, chunks=chunksize, asarray=True)
    coherence_ = da.from_array(coherence, chunks=chunksize, asarray=True)

    # Get a coarse estimate of the unwrapped phase using a low-resolution copy of the
    # interferogram.
    coarse_unwrapped, coarse_conncomp = coarse_unwrap(
        igram=igram_,
        coherence=coherence_,
        nlooks=nlooks,
        unwrap_func=unwrap_func,
        downsample_factor=downsample_factor,
        do_lowpass_filter=do_lowpass_filter,
        shape_factor=shape_factor,
        overhang=overhang,
        ripple=ripple,
        attenuation=attenuation,
    )

    # Unwrap each (high-resolution) tile independently.
    tiled_unwrapped, tiled_conncomp = _util.map_blocks(
        unwrap_func,
        igram_,
        coherence_,
        nlooks=nlooks,
        meta=(np.empty((), dtype=np.float32), np.empty((), dtype=np.uint32)),
    )

    # Add or subtract multiples of 2pi to each connected component to minimize the mean
    # discrepancy between the high-res and low-res unwrapped phase (in order to correct
    # for phase discontinuities between adjacent tiles).
    multiscale_unwrapped = da.map_blocks(
        adjust_conncomp_offset_cycles,
        tiled_unwrapped,
        tiled_conncomp,
        coarse_unwrapped,
        coarse_conncomp,
    )

    # Create the scratch directory if it didn't exist. If no scratch directory was
    # supplied, create a temporary directory that will be cleaned up automatically when
    # the context block is exited.
    with _util.scratch_directory(scratchdir) as d:
        # Create temporary files to store the intermediate connected component labels
        # from coarse & tiled unwrapping.
        coarse_conncomp_tmpfile = unique_binary_file(
            dir=d,
            shape=coarse_conncomp.shape,
            dtype=coarse_conncomp.dtype,
            prefix="coarse_conncomp",
        )
        tiled_conncomp_tmpfile = unique_binary_file(
            dir=d,
            shape=tiled_conncomp.shape,
            dtype=tiled_conncomp.dtype,
            prefix="tiled_conncomp",
        )

        # Store the final unwrapped phase and the intermediate connected component
        # labels. It's necessary to store these intermediate connected component arrays
        # prior to relabeling in order to avoid accidentally executing part of the task
        # graph twice.
        da.store(
            [multiscale_unwrapped, coarse_conncomp, tiled_conncomp],
            [unwrapped, coarse_conncomp_tmpfile, tiled_conncomp_tmpfile],
            lock=_util.get_lock(),
        )

        # Create new Dask arrays from the intermediate connected component arrays that
        # we stored to the disk.
        coarse_conncomp = da.from_array(coarse_conncomp_tmpfile, chunks=chunksize)
        tiled_conncomp = da.from_array(tiled_conncomp_tmpfile, chunks=chunksize)

        # Relabel the tiled connected components based on the coarse components.
        relabeled_conncomp = relabel_hires_conncomps(
            tiled_conncomp,
            coarse_conncomp,
            min_overlap=min_conncomp_overlap,
        )

        # Store the final output connected component labels.
        da.store(relabeled_conncomp, conncomp, lock=_util.get_lock())
