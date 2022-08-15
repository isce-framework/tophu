from typing import Tuple

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

import tophu

from .simulate import simulate_phase_noise, simulate_terrain


def dummy_igram_and_coherence(
    length: int = 128,
    width: int = 128,
) -> Tuple[NDArray[np.complexfloating], NDArray[np.floating]]:
    """
    Return dummy interferogram and coherence arrays (for tests that don't care about
    their values).
    """
    length, width = 128, 128
    igram = np.zeros((length, width), dtype=np.complex64)
    coherence = np.ones((length, width), dtype=np.float32)
    return igram, coherence


def round_to_nearest(n: ArrayLike, base: ArrayLike) -> NDArray:
    """Round to the nearest multiple of `base`."""
    n = np.asanyarray(n)
    base = np.asanyarray(base)
    return base * round(n / base)


def frac_nonzero(arr: ArrayLike) -> float:
    """Compute the fraction of pixels in an array that have nonzero values."""
    return np.count_nonzero(arr) / np.size(arr)


def jaccard_similarity(a, b):
    """
    Compute the Jaccard similarity coefficient (intersect-over-union) of two boolean
    arrays.

    Parameters
    ----------
    a, b : array_like
        The input binary masks.

    Returns
    -------
    J : float
        The Jaccard similarity coefficient of the two inputs.
    """
    return np.sum(a & b) / np.sum(a | b)


class TestMultiScaleUnwrap:
    # XXX Currently we're not testing with PHASS due to an issue where the unwrapped
    # phase seems to not be congruent with the wrapped phase, which causes the
    # processing to fail. This needs to be further investigated.
    @pytest.mark.parametrize("length,width", [(1023, 1023), (1024, 1024)])
    @pytest.mark.parametrize("unwrap", [tophu.ICUUnwrap(), tophu.SnaphuUnwrap()])
    def test_multiscale_unwrap_phase(self, length, width, unwrap):
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
        da = nlooks_az * az_spacing

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
        nlooks = dr * da / (range_res * az_res)

        # Unwrap using the multi-resolution approach.
        unw, conncomp = tophu.multiscale_unwrap(
            igram=igram,
            coherence=coherence,
            nlooks=nlooks,
            unwrap=unwrap,
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
        phasediff = (phase - unw)[mask]
        offset = round_to_nearest(np.mean(phasediff), 2.0 * np.pi)
        good_pixels = np.isclose(unw[mask] + offset, phase[mask], rtol=1e-5, atol=1e-5)
        assert frac_nonzero(good_pixels) > 0.999

        # Check the connected component labels. There should be a single connected
        # component (with label 1) which contains most pixels. Any remaining pixels
        # should be masked out (with label 0).
        unique_labels = set(np.unique(conncomp[mask]))
        assert unique_labels == {1}
        assert frac_nonzero(conncomp) > 0.999

    @pytest.mark.parametrize("unwrap", [tophu.ICUUnwrap(), tophu.SnaphuUnwrap()])
    def test_multiscale_unwrap_phase_conncomps(self, unwrap):
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
        da = nlooks_az * az_spacing

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
        conncomp_mask_1 = np.full((length, width), fill_value=True, dtype=np.bool_)
        conncomp_mask_1[64:-64, 64:-64] = False

        conncomp_mask_2 = np.full((length, width), fill_value=False, dtype=np.bool_)
        conncomp_mask_2[192:-192, 192:-192] = True

        coherence = np.ones((length, width), dtype=np.float32)
        coherence[~conncomp_mask_1 & ~conncomp_mask_2] = 0.01

        # Get effective number of looks.
        nlooks = dr * da / (range_res * az_res)

        # Add phase noise.
        phase += simulate_phase_noise(coherence, nlooks)

        # Form simulated interferogram.
        igram = np.exp(1j * phase)

        # Unwrap using the multi-resolution approach.
        unw, conncomp = tophu.multiscale_unwrap(
            igram=igram,
            coherence=coherence,
            nlooks=nlooks,
            unwrap=unwrap,
            downsample_factor=(3, 3),
            ntiles=(2, 2),
        )

        # Check the unwrapped phase within each expected connected component. The
        # unwrapped phase and absolute (true) phase should differ only by a constant
        # integer multiple of 2pi. The test metric is the fraction of correctly
        # unwrapped pixels, i.e. pixels where the unwrapped phase and absolute phase
        # agree up to some constant number of cycles, excluding masked pixels.
        for mask in [conncomp_mask_1, conncomp_mask_2]:
            phasediff = (phase - unw)[mask]
            offset = round_to_nearest(np.mean(phasediff), 2.0 * np.pi)
            good_pixels = np.isclose(
                unw[mask] + offset, phase[mask], rtol=1e-5, atol=1e-5
            )
            assert frac_nonzero(good_pixels) > 0.999

        # Check the connected component labels.
        # XXX We haven't yet implemented a correction for discrepancies between labels
        # from different tiles. For now, just check the mask of all valid pixels,
        # regardless of their label.
        mask = conncomp != 0
        expected_mask = conncomp_mask_1 | conncomp_mask_2
        assert jaccard_similarity(mask, expected_mask) > 0.99

    def test_shape_mismatch(self):
        length, width = 128, 128
        igram = np.zeros((length, width), dtype=np.complex64)
        coherence = np.ones((length + 1, width + 1), dtype=np.float32)
        errmsg = (
            "shape mismatch: interferogram and coherence arrays must have the same"
            " shape"
        )
        with pytest.raises(ValueError, match=errmsg):
            tophu.multiscale_unwrap(
                igram,
                coherence,
                nlooks=1.0,
                unwrap=tophu.ICUUnwrap(),
                downsample_factor=(3, 3),
                ntiles=(2, 2),
            )

    def test_bad_nlooks(self):
        igram, coherence = dummy_igram_and_coherence()
        with pytest.raises(ValueError, match="effective number of looks must be >= 1"):
            tophu.multiscale_unwrap(
                igram,
                coherence,
                nlooks=0.0,
                unwrap=tophu.ICUUnwrap(),
                downsample_factor=(3, 3),
                ntiles=(2, 2),
            )

    def test_bad_downsample_factor(self):
        igram, coherence = dummy_igram_and_coherence()
        with pytest.raises(ValueError, match="downsample factor must be >= 1"):
            tophu.multiscale_unwrap(
                igram,
                coherence,
                nlooks=1.0,
                unwrap=tophu.ICUUnwrap(),
                downsample_factor=(0, 0),
                ntiles=(2, 2),
            )

    def test_bad_ntiles(self):
        igram, coherence = dummy_igram_and_coherence()
        with pytest.raises(ValueError, match="number of tiles must be >= 1"):
            tophu.multiscale_unwrap(
                igram,
                coherence,
                nlooks=1.0,
                unwrap=tophu.ICUUnwrap(),
                downsample_factor=(3, 3),
                ntiles=(0, 0),
            )
