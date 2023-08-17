import itertools

import dask.array as da
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

        # Output upsampled signal length.
        n_out = ratio * n

        # Upsample.
        output = tophu.upsample_fft(input, out_shape=n_out)

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
            assert np.max(phase_err) < 1e-11

    @pytest.mark.parametrize("ratio", [3, 4])
    @pytest.mark.parametrize("dtype", [float, complex])
    def test_upsample_2d(self, ratio, dtype):
        # Band-limited reference signal.
        bandwidth = 10.0
        c = (1.0 - 1.0j) if (dtype is complex) else 1.0

        def signal(x, y):
            x = np.expand_dims(x, axis=0)
            y = np.expand_dims(y, axis=1)
            return c * np.sinc(bandwidth * x) * np.sinc(bandwidth * y)

        # Evaluate the signal at coarse sample points.
        nx, ny = 64, 65
        samplerate = 2.0 * bandwidth
        x = np.arange(-(nx // 2), (nx + 1) // 2) / samplerate
        y = np.arange(-(ny // 2), (ny + 1) // 2) / samplerate
        input = signal(x, y)

        # Output upsampled signal length.
        nx_out, ny_out = ratio * nx, ratio * ny

        # Upsample.
        output = tophu.upsample_fft(input, out_shape=(ny_out, nx_out))

        # Evaluate the signal at upsampled sample points.
        x_ups = x[0] + ((x[1] - x[0]) / ratio) * np.arange(nx_out)
        y_ups = y[0] + ((y[1] - y[0]) / ratio) * np.arange(ny_out)
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
            assert np.max(phase_err) < 1e-11

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
        output = tophu.upsample_fft(input, out_shape=n)

        # Check that output matches input.
        assert np.allclose(output, input, rtol=1e-12, atol=1e-12)

    def test_nyquist_splitting(self):
        # Input data is real-valued, even-length, with non-zero Nyquist frequency
        # component.
        n = 128
        data = np.zeros(n, dtype=np.float64)
        data[0] = 1.0

        # Check that the upsampled result is purely real-valued.
        n_out = 2 * n
        output_real = tophu.upsample_fft(data, out_shape=n_out)
        output_complex = tophu.upsample_fft(data + 0j, out_shape=n_out)
        assert np.allclose(output_real + 0j, output_complex, rtol=1e-12, atol=1e-12)

    def test_length_two(self):
        # Test input data with length=2.
        data = np.array([1.0 + 0.0j, 0.0 + 0.0j])
        output = tophu.upsample_fft(data, out_shape=4)
        expected = np.array([1.0 + 0.0j, 0.5 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j])
        assert np.allclose(output, expected, rtol=1e-12, atol=1e-12)

    def test_bad_output_ndim(self):
        # Check that `upsample_fft()` fails if the length of `out_shape` doesn't match
        # the input array number of dimensions.
        data = np.zeros((4, 5), dtype=np.complex64)
        errmsg = r"length of out_shape \(3\) must match input array rank \(2\)"
        with pytest.raises(ValueError, match=errmsg):
            tophu.upsample_fft(data, out_shape=(8, 10, 12))

    def test_invalid_out_shape(self):
        # Check that `upsample_fft()` fails if the desired "upsampled" shape was smaller
        # than the input array shape.
        data = np.zeros((4, 5), dtype=np.complex64)
        errmsg = "output shape must be >= input data shape"
        with pytest.raises(ValueError, match=errmsg):
            tophu.upsample_fft(data, out_shape=(3, 4))


class TestUpsampleNearest:
    @pytest.mark.parametrize("order", ["C_CONTIGUOUS", "F_CONTIGUOUS"])
    def test_upsample(self, order):
        input = da.arange(60).reshape(3, 4, 5)

        if order == "F_CONTIGUOUS":
            input = da.asanyarray(input, order="F")

        ratio = (3, 2, 1)
        out_shape = [r * n for (r, n) in zip(ratio, input.shape)]
        output = tophu.upsample_nearest(input, out_shape=out_shape)

        expected = input
        for axis, r in zip(range(input.ndim), ratio):
            expected = da.repeat(expected, repeats=r, axis=axis)

        assert da.allclose(output, expected, rtol=1e-12, atol=1e-12)

    def test_noninteger_upsample_ratio(self):
        # Test upsampling to an output shape that is not an integer multiple of the
        # input array shape.
        input = da.arange(10)
        output = tophu.upsample_nearest(input, out_shape=55)

        # The output sequence should consist of each element from the input array
        # repeated either 5 or 6 times in a row (since the upsampling ratio is 5.5). We
        # use `groupby()` here both to deduplicate consecutive elements and to determine
        # the number of times that each element was repeated consecutively.
        uniq = [k for (k, _) in itertools.groupby(output)]
        assert da.all(input == uniq)

        nrepeats = [len(list(group)) for (_, group) in itertools.groupby(output)]
        assert all(n == 5 for n in nrepeats[:-1])
        assert nrepeats[-1] == 10
