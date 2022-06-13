import numpy as np
import pytest
from isce3.unwrap import snaphu
from numpy.typing import ArrayLike, NDArray
from scipy.stats import mode

import tophu

from .simulate import simulate_phase_noise, simulate_terrain


def round_to_nearest(n: ArrayLike, base: ArrayLike) -> NDArray:
    """Round to the nearest multiple of `base`."""
    n = np.asanyarray(n)
    base = np.asanyarray(base)
    return base * round(n / base)


def frac_nonzero(arr: ArrayLike) -> float:
    """Compute the fraction of pixels in an array that have nonzero values."""
    return np.count_nonzero(arr) / np.size(arr)


class TestUnwrapCallback:
    def test_abstract_class(self):
        # Check that `UnwrapCallback` is an abstract class -- it cannot be directly
        # instantiated.
        with pytest.raises(TypeError, match="Protocols cannot be instantiated"):
            tophu.UnwrapCallback()


class TestSnaphuUnwrap:
    def test_interface(self):
        # Check that `SnaphuUnwrapper` satisfies the interface requirements of
        # `UnwrapCallback`.
        assert issubclass(tophu.SnaphuUnwrap, tophu.UnwrapCallback)

    def test_bad_cost_mode(self):
        with pytest.raises(ValueError, match="unexpected cost mode"):
            tophu.SnaphuUnwrap(cost="asdf")

    def test_bad_init_method(self):
        with pytest.raises(ValueError, match="unexpected initialization method"):
            tophu.SnaphuUnwrap(init_method="asdf")

    def test_unwrap(self):
        # Radar sensor/geometry parameters.
        bperp = 500.0
        altitude = 750_000.0
        near_range = 900_000.0
        range_spacing = 6.25
        az_spacing = 6.0
        range_res = 7.5
        az_res = 6.6
        wvl = 0.24
        transmit_mode = "repeat_pass"
        inc_angle = np.deg2rad(37.0)

        # Multilooking parameters.
        nlooks_range = 5
        nlooks_az = 5
        dr = nlooks_range * range_spacing
        da = nlooks_az * az_spacing

        # Cost mode configuration parameters.
        cost_params = snaphu.TopoCostParams(
            bperp=bperp,
            near_range=near_range,
            dr=dr,
            da=da,
            range_res=range_res,
            az_res=az_res,
            wavelength=wvl,
            transmit_mode=transmit_mode,
            altitude=altitude,
        )

        # Create callback function to run SNAPHU in "topo" cost mode.
        unwrap = tophu.SnaphuUnwrap("topo", cost_params)

        # Multilooked interferogram dimensions.
        length, width = 512, 1024

        # Simulate topographic height map.
        z = simulate_terrain(length, width, scale=4000.0, seed=1234)

        # Simulate expected interferometric phase from topography.
        r = near_range + dr * np.arange(length)
        phase = -4.0 * np.pi / wvl * bperp / r[:, None] * np.sin(inc_angle) * z

        # Correlation coefficient
        corrcoef = np.full((length, width), fill_value=0.9)

        # Get effective number of looks.
        nlooks = dr * da / (range_res * az_res)

        # Add phase noise.
        phase += simulate_phase_noise(corrcoef, nlooks, seed=1234)

        # Create unit-magnitude interferogram.
        igram = np.exp(1j * phase)

        # Unwrap.
        unw, conncomp = unwrap(igram, corrcoef, nlooks)

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
        assert frac_nonzero(good_pixels) > 0.95

        # Check the connected component labels. There should be a single connected
        # component (with label 1) which contains most pixels. Any remaining pixels
        # should be masked out (with label 0).
        unique_labels = set(np.unique(conncomp[mask]))
        assert unique_labels == {1}
        assert frac_nonzero(conncomp) > 0.95


class TestICUUnwrap:
    def test_interface(self):
        # Check that `ICUUnwrapper` satisfies the interface requirements of
        # `UnwrapCallback`.
        assert issubclass(tophu.ICUUnwrap, tophu.UnwrapCallback)

    def test_bad_min_coherence(self):
        errmsg = "minimum coherence must be between 0 and 1"
        with pytest.raises(ValueError, match=errmsg):
            tophu.ICUUnwrap(min_coherence=-0.1)
        with pytest.raises(ValueError, match=errmsg):
            tophu.ICUUnwrap(min_coherence=1.1)

    def test_bad_ntrees(self):
        errmsg = "number of tree realizations must be >= 1"
        with pytest.raises(ValueError, match=errmsg):
            tophu.ICUUnwrap(ntrees=0)

    def test_bad_max_branch_length(self):
        with pytest.raises(ValueError, match="max branch length must be >= 1"):
            tophu.ICUUnwrap(max_branch_length=0)

    def test_bad_phasegrad_window_size(self):
        with pytest.raises(ValueError, match="phase gradient window size must be >= 1"):
            tophu.ICUUnwrap(use_phasegrad_neutrons=True, phasegrad_window_size=0)
        errmsg = "phase gradient window size must be odd-valued"
        with pytest.raises(ValueError, match=errmsg):
            tophu.ICUUnwrap(use_phasegrad_neutrons=True, phasegrad_window_size=4)

    def test_bad_neutron_phasegrad_thresh(self):
        errmsg = "neutron phase gradient threshold must be > 0"
        with pytest.raises(ValueError, match=errmsg):
            tophu.ICUUnwrap(use_phasegrad_neutrons=True, neutron_phasegrad_thresh=0.0)

    def test_bad_neutron_intensity_thresh(self):
        errmsg = "neutron intensity variation threshold must be > 0"
        with pytest.raises(ValueError, match=errmsg):
            tophu.ICUUnwrap(use_intensity_neutrons=True, neutron_intensity_thresh=0.0)

    def test_bad_neutron_coherence_thresh(self):
        errmsg = "neutron coherence threshold must be between 0 and 1"
        with pytest.raises(ValueError, match=errmsg):
            tophu.ICUUnwrap(use_intensity_neutrons=True, neutron_coherence_thresh=-0.1)
        with pytest.raises(ValueError, match=errmsg):
            tophu.ICUUnwrap(use_intensity_neutrons=True, neutron_coherence_thresh=1.1)

    def test_bad_min_conncomp_area_frac(self):
        errmsg = "minimum connected component size must be > 0"
        with pytest.raises(ValueError, match=errmsg):
            tophu.ICUUnwrap(min_conncomp_area_frac=0.0)

    def test_unwrap(self):
        # Multilooked interferogram dimensions.
        length, width = 1024, 256

        # Simulate 2-D absolute phase field with a linear diagonal phase gradient, in
        # radians.
        x = np.linspace(0.0, 50.0, width, dtype=np.float32)
        y = np.linspace(0.0, 100.0, length, dtype=np.float32)
        phase = x + y[:, None]

        # Create masks of connected components.
        mask1 = np.zeros((length, width), dtype=np.bool_)
        mask1[50:450, 25:-25] = True
        mask2 = np.zeros((length, width), dtype=np.bool_)
        mask2[600:, :] = True

        # Simulate correlation coefficient data with islands of high coherence
        # surrounded by low-coherence pixels.
        corrcoef = np.full((length, width), fill_value=0.1, dtype=np.float32)
        corrcoef[mask1 | mask2] = 0.9

        # Add phase noise.
        nlooks = 9.0
        phase += simulate_phase_noise(corrcoef, nlooks, seed=1234)

        # Create unit-magnitude interferogram.
        igram = np.exp(1j * phase)

        # Create callback function to run ICU.
        unwrap = tophu.ICUUnwrap(min_coherence=0.5)

        # Unwrap.
        unw, conncomp = unwrap(igram, corrcoef, nlooks)

        # Check the number of unique nonzero connected component labels.
        unique_labels = set(np.unique(conncomp))
        unique_nonzero_labels = unique_labels - {0}
        assert len(unique_nonzero_labels) == 2

        # Within each masked region, most pixels should be assigned to a single common
        # connected component with nonzero label (some pixels may be mislabeled due to
        # noise). The test checks the fraction of pixels within each masked region whose
        # label matches the modal (most common) label from that region. It also checks
        # that the modal label is nonzero.
        for mask in [mask1, mask2]:
            modal_label, _ = mode(conncomp[mask], axis=None)
            assert frac_nonzero(conncomp[mask] == modal_label) > 0.95
            assert modal_label != 0

        # Check that pixels not belonging to any masked region are labeled zero.
        assert np.all(conncomp[~(mask1 | mask2)] == 0)

        # Check the unwrapped phase within each connected component. The unwrapped phase
        # and absolute (true) phase should differ only by a constant integer multiple of
        # 2pi. The test metric is the fraction of correctly unwrapped pixels, i.e.
        # pixels where the unwrapped phase and absolute phase agree up to some constant
        # number of cycles.
        phasediff = phase - unw
        for label in unique_nonzero_labels:
            mask = conncomp == label
            off = round_to_nearest(np.mean(phasediff[mask]), 2.0 * np.pi)
            good_pixels = np.isclose(unw[mask] + off, phase[mask], rtol=1e-5, atol=1e-5)
            assert frac_nonzero(good_pixels) > 0.95
