from collections import namedtuple

import numpy as np
import pytest
import scipy.signal
from numpy.typing import ArrayLike, NDArray

import tophu


def amp2db(x: ArrayLike) -> NDArray:
    """Convert an amplitude ratio to decibels (dB)."""
    return 20.0 * np.log10(x)


def iseven(n: int) -> bool:
    """Check if an integer is even-valued."""
    return n % 2 == 0


class TestBandpassEquirippleFilter:
    # Helper class to make pytest parameterization more readable.
    params = namedtuple("params", ["shape", "cutoff", "ripple", "attenuation"])

    @pytest.mark.parametrize(
        "shape,cutoff,ripple,attenuation",
        [
            params(shape=1.1, cutoff=0.5, ripple=0.1, attenuation=80.0),
            params(shape=1.2, cutoff=0.2, ripple=1.0, attenuation=60.0),
            params(shape=1.2, cutoff=0.5, ripple=0.5, attenuation=80.0),
            params(shape=1.5, cutoff=0.2, ripple=0.01, attenuation=40.0),
            params(shape=1.5, cutoff=0.5, ripple=0.001, attenuation=80.0),
            params(shape=1.5, cutoff=0.5, ripple=0.1, attenuation=100.0),
            params(shape=3.0, cutoff=0.1, ripple=0.1, attenuation=40.0),
        ],
    )
    @pytest.mark.parametrize("centerfreq", [0.0, 0.5])
    def test_passband_and_stopband_gain(
        self,
        shape,
        cutoff,
        ripple,
        attenuation,
        centerfreq,
    ):
        # Form filter coefficients with the desired characteristics.
        bandwidth = cutoff / shape
        coeffs = tophu.bandpass_equiripple_filter(
            bandwidth,
            shape,
            ripple,
            attenuation,
            centerfreq,
        )

        # Measure the frequency response of the filter.
        f, h = scipy.signal.freqz(coeffs, fs=1.0)
        gain_db = amp2db(np.abs(h))

        # Mask frequency bins in the passband region.
        fmin = centerfreq - 0.5 * bandwidth
        fmax = centerfreq + 0.5 * bandwidth
        passband = (f >= fmin) & (f <= fmax)

        # Mask frequency bins in the stopband region.
        fc1 = centerfreq - 0.5 * cutoff
        fc2 = centerfreq + 0.5 * cutoff
        stopband = (f <= fc1) | (f >= fc2)

        # Check the minimum & maximum gain in the passband, and the maximum gain in the
        # stopband. (Note that the min/max gain in the passband isn't actually symmetric
        # when expressed in dB, but it's close enough for our purposes.)
        assert np.min(gain_db[passband]) >= -1.1 * ripple
        assert np.max(gain_db[passband]) <= 1.1 * ripple
        assert np.max(gain_db[stopband]) <= -0.95 * attenuation

    def test_force_odd_length(self):
        # Check that `force_odd_length=True` results in an odd-length filter in a case
        # where it otherwise wouldn't have been.
        args = 0.25, 1.5, 0.5, 60.0
        for force_odd in [False, True]:
            coeffs = tophu.bandpass_equiripple_filter(*args, force_odd_length=force_odd)
            if force_odd:
                assert not iseven(len(coeffs))
            else:
                assert iseven(len(coeffs))

    @pytest.mark.parametrize("force_odd", [True, False])
    def test_symmetric(self, force_odd):
        # The algorithm should produce only type I & II linear-phase filters. So, for
        # both odd-length and even-length output, check that the filter coefficients are
        # symmetric.
        args = 0.25, 1.5, 0.5, 60.0
        coeffs = tophu.bandpass_equiripple_filter(*args, force_odd_length=force_odd)
        assert np.allclose(coeffs, coeffs[::-1], atol=1e-12, rtol=1e-12)

    def test_bad_bandwidth(self):
        # Check that the function fails when `bandwidth` is invalid.
        shape, ripple, attenuation = 1.5, 0.5, 60.0
        errmsg = "passband width must be > 0 and < 1"
        with pytest.raises(ValueError, match=errmsg):
            tophu.bandpass_equiripple_filter(0.0, shape, ripple, attenuation)
        with pytest.raises(ValueError, match=errmsg):
            tophu.bandpass_equiripple_filter(1.0, shape, ripple, attenuation)

    def test_bad_shape(self):
        # Check that the function fails when `shape` is invalid.
        bandwidth, ripple, attenuation = 0.5, 0.5, 60.0
        errmsg = "shape factor must be > 1 and < 2"
        with pytest.raises(ValueError, match=errmsg):
            tophu.bandpass_equiripple_filter(bandwidth, 1.0, ripple, attenuation)
        with pytest.raises(ValueError, match=errmsg):
            tophu.bandpass_equiripple_filter(bandwidth, 2.0, ripple, attenuation)

    def test_bad_ripple(self):
        # Check that the function fails when `ripple` is invalid.
        bandwidth, shape, attenuation = 0.5, 1.2, 60.0
        with pytest.raises(ValueError, match="passband ripple must be nonzero"):
            tophu.bandpass_equiripple_filter(bandwidth, shape, 0.0, attenuation)

    def test_bad_attenuation(self):
        # Check that the function fails when `attenuation` is invalid.
        bandwidth, shape, ripple = 0.5, 1.2, 0.5
        with pytest.raises(ValueError, match="stopband attenuation must be nonzero"):
            tophu.bandpass_equiripple_filter(bandwidth, shape, ripple, 0.0)

    def test_bad_samplerate(self):
        # Check that the function fails when `samplerate` is invalid.
        args = 0.5, 1.2, 0.5, 60.0
        with pytest.raises(ValueError):
            tophu.bandpass_equiripple_filter(*args, samplerate=0.0)

    def test_bad_maxiter(self):
        # Check that the function fails when `maxiter` is invalid.
        args = 0.5, 1.2, 0.5, 60.0
        with pytest.raises(ValueError, match="max number of iterations must be >= 1"):
            tophu.bandpass_equiripple_filter(*args, maxiter=0)

    def test_bad_grid_density(self):
        # Check that the function fails when `grid_density` is invalid.
        args = 0.5, 1.2, 0.5, 60.0
        with pytest.raises(ValueError, match="grid density must be >= 1"):
            tophu.bandpass_equiripple_filter(*args, grid_density=0)
