from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

import tophu
from tophu import UnwrapCallback

from .simulate import simulate_phase_noise, simulate_terrain

UNWRAP_FUNCS: list[UnwrapCallback] = [
    # tophu.ICUUnwrap(),
    tophu.PhassUnwrap(),
    tophu.SnaphuUnwrap(),
]


def dummy_inputs_and_outputs(
    length: int = 128,
    width: int = 128,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.unsignedinteger],
    NDArray[np.complexfloating],
    NDArray[np.floating],
]:
    """
    Return dummy interferogram and coherence arrays (for tests that don't care about
    their values).
    """
    unwrapped = np.zeros((length, width), dtype=np.float32)
    conncomp = np.zeros((length, width), dtype=np.uint32)
    igram = np.zeros((length, width), dtype=np.complex64)
    coherence = np.ones((length, width), dtype=np.float32)
    return unwrapped, conncomp, igram, coherence


def round_to_nearest(n: ArrayLike, base: ArrayLike) -> NDArray:
    """Round to the nearest multiple of `base`."""
    n = np.asanyarray(n)
    base = np.asanyarray(base)
    return base * round(n / base)


def frac_nonzero(arr: ArrayLike) -> float:
    """Compute the fraction of pixels in an array that have nonzero values."""
    return np.count_nonzero(arr) / np.size(arr)


def jaccard_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """
    Compute the Jaccard similarity coefficient (intersect-over-union) of two boolean
    arrays.

    Parameters
    ----------
    a, b : numpy.ndarray
        The input binary masks.

    Returns
    -------
    J : float
        The Jaccard similarity coefficient of the two inputs.
    """
    a = np.asanyarray(a, dtype=np.bool_)
    b = np.asanyarray(b, dtype=np.bool_)
    return np.sum(a & b) / np.sum(a | b)


def list_subdirs(dir: str | os.PathLike) -> list[Path]:
    """
    List all immediate subdirectories of the specified directory.

    Parameters
    ----------
    dir : path-like
        The parent directory. Must be an existing file system directory.

    Returns
    -------
    subdirs : list of pathlib.Path
        All subdirectories of `dir`.
    """
    dir = Path(dir)

    if not dir.is_dir():
        raise ValueError(f"'{dir}' is not an existing directory.")

    return [d for d in dir.iterdir() if d.is_dir()]


class TestMultiScaleUnwrap:
    @pytest.mark.parametrize("length,width", [(1023, 1023), (1024, 1024)])
    @pytest.mark.parametrize("unwrap_func", UNWRAP_FUNCS)
    def test_multiscale_unwrap_phase(
        self,
        length: int,
        width: int,
        unwrap_func: UnwrapCallback,
    ):
        # Radar sensor/geometry parameters.
        near_range = 900_000.0
        range_spacing = 6.25
        az_spacing = 6.0
        range_res = 7.5
        az_res = 6.6
        bperp = 500.0
        wvl = 0.24
        inc_angle = np.deg2rad(37.0)

        # Simulate random topography.
        z = simulate_terrain(length, width, scale=2000.0, smoothness=0.9, seed=123)

        # Get multilooked sample spacing.
        nlooks_range = 5
        nlooks_az = 5
        dr = nlooks_range * range_spacing
        daz = nlooks_az * az_spacing

        # Simulate topographic phase term.
        r = near_range + dr * np.arange(width)
        phase = -4.0 * np.pi / wvl * bperp / r[None, :] * np.sin(inc_angle) * z

        # Add a diagonal linear phase gradient such that, if we were to naively unwrap
        # by tiles without applying a post-processing correction, each tile will have
        # some relative phase offset with respect to the other tiles, resulting in
        # discontinuities at the borders between tiles.
        x = np.linspace(0.0, 50.0, width, dtype=np.float32)
        y = np.linspace(0.0, 50.0, length, dtype=np.float32)
        phase += x + y[:, None]

        # Form simulated interferogram & coherence with no noise.
        igram = np.exp(1j * phase)
        coherence = np.ones((length, width), dtype=np.float32)

        # Get effective number of looks.
        nlooks = dr * daz / (range_res * az_res)

        # Init output arrays.
        unwrapped = np.zeros((length, width), dtype=np.float32)
        conncomp = np.zeros((length, width), dtype=np.uint32)

        # Unwrap using the multi-resolution approach.
        tophu.multiscale_unwrap(
            unwrapped=unwrapped,
            conncomp=conncomp,
            igram=igram,
            coherence=coherence,
            nlooks=nlooks,
            unwrap_func=unwrap_func,
            downsample_factor=(3, 3),
            ntiles=(2, 2),
        )

        # Get a mask of valid pixels (pixels that were assigned to some connected
        # component).
        mask = conncomp != 0

        # Check the unwrapped phase. The unwrapped phase and absolute (true) phase
        # should differ only by a constant integer multiple of 2pi. The test metric is
        # the fraction of correctly unwrapped pixels, i.e. pixels where the unwrapped
        # phase and absolute phase agree up to some constant number of cycles, excluding
        # masked pixels.
        phasediff = (phase - unwrapped)[mask]
        offset = round_to_nearest(np.mean(phasediff), 2.0 * np.pi)
        good_pixels = np.isclose(
            unwrapped[mask] + offset, phase[mask], rtol=1e-5, atol=1e-5
        )
        assert frac_nonzero(good_pixels) > 0.999

        # Check the connected component labels. There should be a single connected
        # component (with label 1) which contains most pixels. Any remaining pixels
        # should be masked out (with label 0).
        unique_labels = set(np.unique(conncomp[mask]))
        assert unique_labels == {1}
        assert frac_nonzero(conncomp) > 0.999

    @pytest.mark.parametrize("downsample_factor", [(2, 2), (3, 3)])
    @pytest.mark.parametrize("unwrap_func", UNWRAP_FUNCS)
    def test_multiscale_unwrap_phase_conncomps(
        self,
        downsample_factor: tuple[int, int],
        unwrap_func: UnwrapCallback,
    ):
        length, width = 512, 512

        # Radar sensor/geometry parameters.
        near_range = 900_000.0
        range_spacing = 6.25
        az_spacing = 6.0
        range_res = 7.5
        az_res = 6.6
        bperp = 500.0
        wvl = 0.24
        inc_angle = np.deg2rad(37.0)

        # Simulate random topography.
        z = simulate_terrain(length, width, scale=2000.0, smoothness=0.9, seed=123)

        # Get multilooked sample spacing.
        nlooks_range = 5
        nlooks_az = 5
        dr = nlooks_range * range_spacing
        daz = nlooks_az * az_spacing

        # Simulate topographic phase term.
        r = near_range + dr * np.arange(width)
        phase = -4.0 * np.pi / wvl * bperp / r[None, :] * np.sin(inc_angle) * z

        # Add a diagonal linear phase gradient such that, if we were to naively unwrap
        # by tiles without applying a post-processing correction, each tile will have
        # some relative phase offset with respect to the other tiles, resulting in
        # discontinuities at the borders between tiles.
        x = np.linspace(0.0, 50.0, width, dtype=np.float32)
        y = np.linspace(0.0, 50.0, length, dtype=np.float32)
        phase += x + y[:, None]

        # Form two islands of high coherence that span multiple tiles, separated by low
        # coherence pixels.
        region1_mask = np.full((length, width), fill_value=True, dtype=np.bool_)
        region1_mask[64:-64, 64:-64] = False

        region2_mask = np.full((length, width), fill_value=False, dtype=np.bool_)
        region2_mask[192:-192, 192:-192] = True

        coherence = np.ones((length, width), dtype=np.float32)
        coherence[~region1_mask & ~region2_mask] = 0.01

        # Get effective number of looks.
        nlooks = dr * daz / (range_res * az_res)

        # Add phase noise.
        phase += simulate_phase_noise(coherence, nlooks)

        # Form simulated interferogram.
        igram = np.exp(1j * phase)

        # Init output arrays.
        unwrapped = np.zeros((length, width), dtype=np.float32)
        conncomp = np.zeros((length, width), dtype=np.uint32)

        # Unwrap using the multi-resolution approach.
        tophu.multiscale_unwrap(
            unwrapped=unwrapped,
            conncomp=conncomp,
            igram=igram,
            coherence=coherence,
            nlooks=nlooks,
            unwrap_func=unwrap_func,
            downsample_factor=downsample_factor,
            ntiles=(2, 2),
        )

        # Get a mask of valid pixels (pixels that were assigned to some connected
        # component).
        valid_mask = conncomp != 0

        # Check the unwrapped phase within each expected connected component. The
        # unwrapped phase and absolute (true) phase should differ only by a constant
        # integer multiple of 2pi. The test metric is the fraction of correctly
        # unwrapped pixels, i.e. pixels where the unwrapped phase and absolute phase
        # agree up to some constant number of cycles, excluding masked pixels.
        for region_mask in [region1_mask, region2_mask]:
            mask = region_mask & valid_mask
            phasediff = (phase - unwrapped)[mask]
            offset = round_to_nearest(np.mean(phasediff), 2.0 * np.pi)
            good_pixels = np.isclose(
                unwrapped[mask] + offset, phase[mask], rtol=1e-5, atol=1e-5
            )
            assert frac_nonzero(good_pixels) > 0.999

        # Check the set of unique connected component labels. There should be two
        # connected components, labeled 1 and 2, as well as masked-out pixels labeled 0.
        assert set(np.unique(conncomp)) == {0, 1, 2}

        # Check that each high-coherence region is associated with a single connected
        # component. The test checks the fraction of pixels within each region mask
        # whose label matches the modal (most common) label from that region. It also
        # checks that the modal label is nonzero.
        for mask in [region1_mask, region2_mask]:
            modal_label, _ = tophu.mode(conncomp[mask])
            assert frac_nonzero(conncomp[mask] == modal_label) > 0.95
            assert modal_label != 0

        # Check that the modal label of each region mask is different.
        assert tophu.mode(conncomp[region1_mask]) != tophu.mode(conncomp[region2_mask])

        # Check that the remaining non-region pixels are masked out (labeled 0). The
        # test compares the fraction of non-region pixels that were labeled 0 to a
        # predefined threshold.
        nonregion_mask = ~(region1_mask | region2_mask)
        assert frac_nonzero(conncomp[nonregion_mask] == 0) > 0.95

    @pytest.mark.parametrize("downsample_factor", [(1, 1), (1, 4), (5, 1)])
    def test_multiscale_unwrap_single_look(self, downsample_factor: tuple[int, int]):
        length, width = map(lambda d: 256 * d, downsample_factor)

        # Radar sensor/geometry parameters.
        near_range = 900_000.0
        range_spacing = 6.25
        az_spacing = 6.0
        range_res = 7.5
        az_res = 6.6
        bperp = 500.0
        wvl = 0.24
        inc_angle = np.deg2rad(37.0)

        # Simulate random topography.
        z = simulate_terrain(length, width, scale=2000.0, smoothness=0.9, seed=123)

        # Get multilooked sample spacing.
        nlooks_range = 5
        nlooks_az = 5
        dr = nlooks_range * range_spacing
        daz = nlooks_az * az_spacing

        # Simulate topographic phase term.
        r = near_range + dr * np.arange(width)
        phase = -4.0 * np.pi / wvl * bperp / r[None, :] * np.sin(inc_angle) * z

        # Add a diagonal linear phase gradient such that, if we were to naively unwrap
        # by tiles without applying a post-processing correction, each tile will have
        # some relative phase offset with respect to the other tiles, resulting in
        # discontinuities at the borders between tiles.
        x = np.linspace(0.0, 50.0, width, dtype=np.float32)
        y = np.linspace(0.0, 50.0, length, dtype=np.float32)
        phase += x + y[:, None]

        # Form simulated interferogram & coherence with no noise.
        igram = np.exp(1j * phase)
        coherence = np.ones((length, width), dtype=np.float32)

        # Get effective number of looks.
        nlooks = dr * daz / (range_res * az_res)

        # Init output arrays.
        unwrapped = np.zeros((length, width), dtype=np.float32)
        conncomp = np.zeros((length, width), dtype=np.uint32)

        # Unwrap using the multi-resolution approach.
        tophu.multiscale_unwrap(
            unwrapped=unwrapped,
            conncomp=conncomp,
            igram=igram,
            coherence=coherence,
            nlooks=nlooks,
            unwrap_func=tophu.PhassUnwrap(),
            downsample_factor=downsample_factor,
            ntiles=downsample_factor,
        )

        # Get a mask of valid pixels (pixels that were assigned to some connected
        # component).
        mask = conncomp != 0

        # Check the unwrapped phase. The unwrapped phase and absolute (true) phase
        # should differ only by a constant integer multiple of 2pi. The test metric is
        # the fraction of correctly unwrapped pixels, i.e. pixels where the unwrapped
        # phase and absolute phase agree up to some constant number of cycles, excluding
        # masked pixels.
        phasediff = (phase - unwrapped)[mask]
        offset = round_to_nearest(np.mean(phasediff), 2.0 * np.pi)
        good_pixels = np.isclose(
            unwrapped[mask] + offset, phase[mask], rtol=1e-5, atol=1e-5
        )
        assert frac_nonzero(good_pixels) > 0.999

        # Check the connected component labels. There should be a single connected
        # component (with label 1) which contains most pixels. Any remaining pixels
        # should be masked out (with label 0).
        unique_labels = set(np.unique(conncomp[mask]))
        assert unique_labels == {1}
        assert frac_nonzero(conncomp) > 0.99

    def test_unw_shape_mismatch(self):
        length, width = 128, 128
        _, conncomp, igram, coherence = dummy_inputs_and_outputs(length, width)
        unwrapped = np.zeros((length + 1, width + 1), dtype=np.float32)
        errmsg = "shape mismatch: igram and unwrapped must have the same shape"
        with pytest.raises(ValueError, match=errmsg):
            tophu.multiscale_unwrap(
                unwrapped,
                conncomp,
                igram,
                coherence,
                nlooks=1.0,
                unwrap_func=tophu.SnaphuUnwrap(),
                downsample_factor=(3, 3),
                ntiles=(2, 2),
            )

    def test_conncomp_shape_mismatch(self):
        length, width = 128, 128
        unwrapped, _, igram, coherence = dummy_inputs_and_outputs(length, width)
        conncomp = np.zeros((length + 1, width + 1), dtype=np.uint32)
        errmsg = "shape mismatch: unwrapped and conncomp must have the same shape"
        with pytest.raises(ValueError, match=errmsg):
            tophu.multiscale_unwrap(
                unwrapped,
                conncomp,
                igram,
                coherence,
                nlooks=1.0,
                unwrap_func=tophu.SnaphuUnwrap(),
                downsample_factor=(3, 3),
                ntiles=(2, 2),
            )

    def test_coherence_shape_mismatch(self):
        length, width = 128, 128
        unwrapped, conncomp, igram, _ = dummy_inputs_and_outputs(length, width)
        coherence = np.ones((length + 1, width + 1), dtype=np.float32)
        errmsg = "shape mismatch: igram and coherence must have the same shape"
        with pytest.raises(ValueError, match=errmsg):
            tophu.multiscale_unwrap(
                unwrapped,
                conncomp,
                igram,
                coherence,
                nlooks=1.0,
                unwrap_func=tophu.SnaphuUnwrap(),
                downsample_factor=(3, 3),
                ntiles=(2, 2),
            )

    def test_bad_igram_ndim(self):
        shape = (2, 128, 128)
        igram = np.zeros(shape, dtype=np.complex64)
        coherence = np.ones(shape, dtype=np.float32)
        unwrapped = np.zeros(shape, dtype=np.complex64)
        conncomp = np.zeros(shape, dtype=np.float32)
        with pytest.raises(ValueError, match="input array must be 2-dimensional"):
            tophu.multiscale_unwrap(
                unwrapped,
                conncomp,
                igram,
                coherence,
                nlooks=1.0,
                unwrap_func=tophu.SnaphuUnwrap(),
                downsample_factor=(3, 3),
                ntiles=(2, 2, 2),
            )

    def test_bad_nlooks(self):
        unwrapped, conncomp, igram, coherence = dummy_inputs_and_outputs()
        with pytest.raises(ValueError, match="effective number of looks must be >= 1"):
            tophu.multiscale_unwrap(
                unwrapped,
                conncomp,
                igram,
                coherence,
                nlooks=0.0,
                unwrap_func=tophu.SnaphuUnwrap(),
                downsample_factor=(3, 3),
                ntiles=(2, 2),
            )

    def test_bad_downsample_factor(self):
        unwrapped, conncomp, igram, coherence = dummy_inputs_and_outputs()
        with pytest.raises(ValueError, match="downsample factor must be >= 1"):
            tophu.multiscale_unwrap(
                unwrapped,
                conncomp,
                igram,
                coherence,
                nlooks=1.0,
                unwrap_func=tophu.SnaphuUnwrap(),
                downsample_factor=(0, 0),
                ntiles=(2, 2),
            )

    def test_bad_ntiles(self):
        unwrapped, conncomp, igram, coherence = dummy_inputs_and_outputs()
        with pytest.raises(ValueError, match="number of tiles must be >= 1"):
            tophu.multiscale_unwrap(
                unwrapped,
                conncomp,
                igram,
                coherence,
                nlooks=1.0,
                unwrap_func=tophu.ICUUnwrap(),
                downsample_factor=(3, 3),
                ntiles=(0, 0),
            )

    @pytest.mark.parametrize("overhang", [-0.1, 1.1])
    def test_bad_overhang(self, overhang: float):
        unwrapped, conncomp, igram, coherence = dummy_inputs_and_outputs()
        with pytest.raises(ValueError, match="overhang must be between 0 and 1"):
            tophu.multiscale_unwrap(
                unwrapped,
                conncomp,
                igram,
                coherence,
                nlooks=1.0,
                unwrap_func=tophu.SnaphuUnwrap(),
                downsample_factor=(3, 3),
                ntiles=(2, 2),
                overhang=overhang,
            )

    def test_scratchdir(self):
        unwrapped, conncomp, igram, coherence = dummy_inputs_and_outputs()

        with TemporaryDirectory() as d:
            scratchdir = Path(d)

            tophu.multiscale_unwrap(
                unwrapped,
                conncomp,
                igram,
                coherence,
                nlooks=1.0,
                unwrap_func=tophu.SnaphuUnwrap(),
                downsample_factor=(3, 3),
                ntiles=(2, 2),
                scratchdir=scratchdir,
            )

            # Check that the scratch directory still exists.
            assert scratchdir.is_dir()

            # The scratch directory should contain 5 unique subdirectories -- one for
            # each tile and one for the coarse unwrapping step.
            subdirs = list_subdirs(scratchdir)
            assert len(subdirs) == 5
