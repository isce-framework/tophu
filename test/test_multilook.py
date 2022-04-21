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

import warnings

import numpy as np
import pytest

import tophu


class TestMultilook:
    def test_multilook_1d(self):
        # Expected output array.
        expected = np.arange(4, dtype=np.float64)

        # Build the input array by repeating each element `nlooks` times.
        nlooks = 11
        input = np.repeat(expected, repeats=nlooks)

        # Multilook.
        output = tophu.multilook(input, nlooks=nlooks)

        # Check results.
        assert output.shape == expected.shape
        assert output.dtype == expected.dtype
        assert np.allclose(output, expected, rtol=1e-12, atol=1e-12)

    def test_multilook_2d(self):
        # Expected output array.
        expected = np.arange(12, dtype=np.float64).reshape(3, 4)

        # Build the input array by repeating each element `nlooks` times.
        nlooks = (7, 5)
        input = expected
        for i, n in enumerate(nlooks):
            input = np.repeat(input, repeats=n, axis=i)

        # Multilook.
        output = tophu.multilook(input, nlooks=nlooks)

        # Check results.
        assert output.shape == expected.shape
        assert output.dtype == expected.dtype
        assert np.allclose(output, expected, rtol=1e-12, atol=1e-12)

    def test_multilook_3d(self):
        # Expected output array.
        expected = np.arange(60, dtype=np.float64).reshape(3, 4, 5)

        # Build the input array by repeating each element `nlooks` times.
        nlooks = (5, 1, 3)
        input = expected
        for i, n in enumerate(nlooks):
            input = np.repeat(input, repeats=n, axis=i)

        # Multilook.
        output = tophu.multilook(input, nlooks=nlooks)

        # Check results.
        assert output.shape == expected.shape
        assert output.dtype == expected.dtype
        assert np.allclose(output, expected, rtol=1e-12, atol=1e-12)

    def test_nlooks_length_mismatch(self):
        # Check that `multilook()` fails if length of `nlooks` doesn't match `arr.ndim`.
        arr = np.zeros((15, 15), dtype=np.float64)
        with pytest.raises(ValueError, match="length mismatch"):
            tophu.multilook(arr, nlooks=(1, 2, 3))

    def test_zero_or_negative_nlooks(self):
        # Check that `multilook()` fails if `nlooks` has zero or negative values.
        arr = np.zeros((15, 15), dtype=np.float64)
        with pytest.raises(ValueError, match="number of looks must be >= 1"):
            tophu.multilook(arr, nlooks=(-1, 3))
        with pytest.raises(ValueError, match="number of looks must be >= 1"):
            tophu.multilook(arr, nlooks=(5, 0))

    def test_nlooks_too_large(self):
        # Check that `multilook()` fails if `nlooks` is larger than the input array
        # shape (along any axis).
        arr = np.zeros((15, 15), dtype=np.float64)
        errmsg = "number of looks should not exceed array shape"
        with pytest.raises(ValueError, match=errmsg):
            tophu.multilook(arr, nlooks=(1, 16))

    def test_throwaway_samples_warning(self):
        # Check that a warning is emitted if there are throwaway samples due to
        # the input array shape not being an interger multiple of `nlooks`.
        arr = np.zeros((21, 21), dtype=np.float64)
        with warnings.catch_warnings(record=True) as w:
            # Run `multilook()` with all warnings enabled.
            warnings.simplefilter("always")
            output = tophu.multilook(arr, nlooks=(4, 5))

            # Check that a single warning was emitted.
            assert len(w) == 1

            # Check the warning category and message.
            assert issubclass(w[0].category, RuntimeWarning)
            assert "..." in str(w[0].message)

        # Check the output shape.
        assert output.shape == (5, 4)
