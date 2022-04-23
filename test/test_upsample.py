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

import numpy as np
import pytest
from numpy.typing import NDArray

import tophu


def phasediff(a: NDArray, b: NDArray) -> NDArray:
    # Mask zero-magnitude values since the phase angle is not meaningful.
    eps = 1e-12
    mask = (np.abs(a) > eps) & (np.abs(b) > eps)

    return mask * np.angle(a * b.conj())


class TestUpsampleFFT:
    @pytest.mark.parametrize("n", [128, 129])
    @pytest.mark.parametrize("ratio", [2, 5])
    @pytest.mark.parametrize("dtype", [float, complex])
    def test_upsample_1d(self, n, ratio, dtype):
        # Band-limited reference signal.
        bandwidth = 10.0
        c = (1.0 - 1.0j) if (dtype is complex) else 1.0

        def signal(t: NDArray) -> NDArray:
            return c * np.sinc(bandwidth * t)

        # Evaluate the signal at coarse sample points.
        samplerate = 2.0 * bandwidth
        t = np.arange(-(n // 2), (n + 1) // 2) / samplerate
        input = signal(t)

        # Upsample.
        output = tophu.upsample(input, ratio=ratio, method="fft")

        # Evaluate the signal at upsampled sample points.
        t0 = t[0]
        dt = t[1] - t[0]
        t_ups = t0 + (dt / ratio) * np.arange(ratio * n)
        expected = signal(t_ups)

        # Check output shape & dtype.
        assert output.shape == expected.shape
        assert output.dtype == dtype

        # Check that decimated upsampled signal matches input signal
        assert np.allclose(output[::ratio], input, rtol=1e-12, atol=1e-12)

        # Check absolute error in the middle 80% & 50% of samples.
        abs_err = np.abs(output - expected)
        n_ups = ratio * n
        middle_80pct = slice(n_ups // 10, -n_ups // 10)
        middle_50pct = slice(n_ups // 4, -n_ups // 4)
        assert np.max(abs_err[middle_80pct]) < 1e-2
        assert np.max(abs_err[middle_50pct]) < 1e-3

        if dtype is complex:
            # Check phase error.
            phase_err = phasediff(output, expected)
            assert np.max(phase_err) < 1e-12

    @pytest.mark.parametrize("ratio", [3, 4])
    @pytest.mark.parametrize("dtype", [float, complex])
    def test_upsample_2d(self, ratio, dtype):
        # Band-limited reference signal.
        bandwidth = 10.0
        c = (1.0 - 1.0j) if (dtype is complex) else 1.0

        def signal(x, y):
            x = np.expand_dims(x, axis=1)
            y = np.expand_dims(y, axis=0)
            return c * np.sinc(bandwidth * x) * np.sinc(bandwidth * y)

        # Evaluate the signal at coarse sample points.
        nx, ny = 64, 65
        samplerate = 2.0 * bandwidth
        x = np.arange(-(nx // 2), (nx + 1) // 2) / samplerate
        y = np.arange(-(ny // 2), (ny + 1) // 2) / samplerate
        input = signal(x, y)

        # Upsample.
        output = tophu.upsample(input, ratio=ratio, method="fft")

        # Evaluate the signal at upsampled sample points.
        x_ups = x[0] + ((x[1] - x[0]) / ratio) * np.arange(ratio * nx)
        y_ups = y[0] + ((y[1] - y[0]) / ratio) * np.arange(ratio * ny)
        expected = signal(x_ups, y_ups)

        # Check output shape & dtype.
        assert output.shape == expected.shape
        assert output.dtype == dtype

        # Check that decimated upsampled signal matches input signal
        assert np.allclose(output[::ratio, ::ratio], input, rtol=1e-12, atol=1e-12)

        # Check absolute error in the middle 80% & 50% of samples.
        abs_err = np.abs(output - expected)
        middle_80pct = tuple([slice(s // 10, -s // 10) for s in abs_err.shape])
        middle_50pct = tuple([slice(s // 4, -s // 4) for s in abs_err.shape])
        assert np.max(abs_err[middle_80pct]) < 1e-2
        assert np.max(abs_err[middle_50pct]) < 1e-3

        if dtype is complex:
            # Check phase error.
            phase_err = phasediff(output, expected)
            assert np.max(phase_err) < 1e-12

    def test_identity(self):
        # Reference signal.
        def signal(t: NDArray) -> NDArray:
            return np.cos(2.0 * np.pi * t)

        # Evaluate the signal at coarse sample points.
        n = 51
        samplerate = 20.0
        t = np.arange(-(n // 2), (n + 1) // 2) / samplerate
        input = signal(t)

        # "Upsample" with ratio=1.
        output = tophu.upsample(input, ratio=1, method="fft")

        # Check that output matches input.
        assert np.allclose(output, input, rtol=1e-12, atol=1e-12)

    def test_nyquist_splitting(self):
        # Input data is real-valued, even-length, with non-zero Nyquist frequency
        # component.
        data = np.zeros(128, dtype=np.float64)
        data[0] = 1.0

        # Check that the upsampled result is purely real-valued.
        output_real = tophu.upsample(data, ratio=2, method="fft")
        output_complex = tophu.upsample(data + 0j, ratio=2, method="fft")
        assert np.allclose(output_real + 0j, output_complex, rtol=1e-12, atol=1e-12)

    def test_axis_order(self):
        # Check the output shape when ordering of `axes` is non-sequential.
        data = np.zeros((3, 4, 5), dtype=np.complex64)
        output = tophu.upsample(data, ratio=(2, 3), axes=(2, 0), method="fft")
        assert output.shape == (9, 4, 10)

    def test_ratio_axes_length_mismatch(self):
        # Check that `upsample()` fails if lengths of `ratio` and `axes` are
        # inconsistent.
        data = np.zeros((4, 5), dtype=np.complex64)
        errmsg = r"length of ratio \(2\) must match number of upsample axes \(1\)"
        with pytest.raises(ValueError, match=errmsg):
            tophu.upsample(data, ratio=(2, 3), axes=0, method="fft")

    def test_invalid_ratio(self):
        # Check that `upsample()` fails if `ratio` is < 1.
        data = np.zeros((4, 5), dtype=np.complex64)
        with pytest.raises(ValueError, match="upsample ratio must be >= 1"):
            tophu.upsample(data, ratio=(2, 0), method="fft")

    def test_invalid_axes(self):
        # Check that `upsample()` fails if any axis is out of bounds.
        data = np.zeros((4, 5), dtype=np.complex64)
        errmsg = "axis 2 is out of bounds for array of dimension 2"
        with pytest.raises(np.AxisError, match=errmsg):
            tophu.upsample(data, ratio=2, axes=(1, 2), method="fft")

    def test_duplicate_axes(self):
        # Check that `upsample()` fails if any axis in `axes` is repeated.
        data = np.zeros((3, 4, 5), dtype=np.complex64)
        with pytest.raises(ValueError, match="repeated axis"):
            tophu.upsample(data, ratio=3, axes=(2, 2), method="fft")
        with pytest.raises(ValueError, match="repeated axis"):
            tophu.upsample(data, ratio=3, axes=(2, -1), method="fft")
