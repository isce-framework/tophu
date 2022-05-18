import numpy as np
import pytest
from isce3.unwrap import snaphu
from numpy.typing import ArrayLike

import tophu

from .simulate import simulate_phase_noise, simulate_terrain


class TestUnwrapCallback:
    def test_abstract_class(self):
        # Check that `UnwrapCallback` is an abstract class -- it cannot be directly
        # instantiated.
        with pytest.raises(TypeError, match="Protocols cannot be instantiated"):
            tophu.UnwrapCallback()


class TestSnaphuUnwrapper:
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

    def test_snaphu(self):
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

        # Rounds a number to the nearest multiple of `base`.
        def round_to_nearest(n, base):
            return base * round(n / base)

        # Computes the fraction of pixels in an array that have nonzero values.
        def frac_nonzero(arr: ArrayLike) -> float:
            return np.count_nonzero(arr) / np.size(arr)

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
