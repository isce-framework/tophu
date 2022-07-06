from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "simulate_phase_noise",
    "simulate_terrain",
]


def simulate_phase_noise(
    corrcoef: ArrayLike,
    nlooks: float,
    *,
    seed: Optional[int] = None,
) -> NDArray:
    r"""Simulate multilooked interferogram phase noise samples.

    Generate pseudo-random noise samples that approximately match the expected
    distribution of multilooked interferogram phase.

    The resulting samples are zero-mean Gaussian distributed, with variance equal to the
    Cramer-Rao bound of the Maximum Likelihood Estimator for the interferometric
    phase\ :footcite:p:`rodriguez:1992`. This simple approximation is most accurate for
    high coherence and large number of looks. The true phase difference distribution is
    more complicated\ :footcite:p:`lee:1994`.

    Parameters
    ----------
    corrcoef : array_like
        Interferometric correlation coefficient.
    nlooks : float
        Number of independent looks.
    seed : int or None, optional
        Seed for initializing pseudo-random number generator state. Must be nonnegative.
        If None, then the generator will be initialized randomly. (default: None)

    Returns
    -------
    phi : array_like
        Phase noise samples.
    """
    corrcoef = np.asanyarray(corrcoef)

    # Setup a pseudo-random number generator.
    rng = np.random.default_rng(seed)

    # Approximate interferometric phase standard deviation using a simple
    # approximation that holds for high coherence/number of looks.
    sigma = 1.0 / np.sqrt(2.0 * nlooks) * np.sqrt(1.0 - corrcoef**2) / corrcoef

    # Generate zero-mean Gaussian-distributed phase noise samples.
    return rng.normal(scale=sigma)


def simulate_terrain(
    length: int,
    width: int,
    *,
    scale: float = 1000.0,
    smoothness: float = 0.8,
    seed: Optional[int] = None,
) -> NDArray:
    r"""Simulate topography using the Diamond-Square algorithm.

    Generate a 2-D fractal height map using the Diamond-Square algorithm\
    :footcite:p:`miller:1986`.

    Parameters
    ----------
    length, width : int
        Output dimensions.
    scale : float
        Controls the range of variation in height. Must be positive. (default: 1000.0)
    smoothness : float
        Smoothness constant. Must be in the range [0, 1]. Larger values yield smoother
        terrain. Smaller values result in more jagged terrain. (default: 0.8)
    seed : int or None, optional
        Seed for initializing pseudo-random number generator state. Must be nonnegative.
        If None, then the generator will be initialized randomly. (default: None)

    Returns
    -------
    z : numpy.ndarray
        Height map.
    """
    # Validate inputs.
    if (length <= 0) or (width <= 0):
        raise ValueError("output array dimensions must be positive")
    if scale <= 0.0:
        raise ValueError("scale factor must be positive")
    if not (0.0 <= smoothness <= 1.0):
        raise ValueError("smoothness constant must be between 0.0 and 1.0")

    # Setup a pseudo-random number generator.
    rng = np.random.default_rng(seed)

    # For positive n, returns the smallest power of two that is >= n
    def next_power_of_two(n):
        return 1 << (n - 1).bit_length()

    # The algorithm operates on a square array with length,width = 2^n + 1
    size = next_power_of_two(max(length, width) - 1) + 1
    z = np.zeros((size, size))

    # Proceed with alternating iterations of "diamond" and "square" steps with
    # progressively smaller strides. The stride is halved at each iteration until, in
    # the last iteration, it's 2 pixels.
    stride = size - 1
    while stride > 1:
        # In the diamond step, the pixel at the midpoint of each square is set to the
        # average value of the square's four corner pixels plus a random offset.

        # Compute the average of the four corner pixels for each square.
        stop = size - stride
        uli, ulj = np.ogrid[0:stop:stride, 0:stop:stride]
        uri, urj = np.ogrid[0:stop:stride, stride:size:stride]
        lli, llj = np.ogrid[stride:size:stride, 0:stop:stride]
        lri, lrj = np.ogrid[stride:size:stride, stride:size:stride]
        avg = 0.25 * (z[uli, ulj] + z[uri, urj] + z[lli, llj] + z[lri, lrj])

        # Set the midpoint pixel to the average of the four corner pixels plus some
        # random value.
        start = stride // 2
        di, dj = np.ogrid[start:size:stride, start:size:stride]
        rval = scale * rng.uniform(-0.5, 0.5, size=(z[di, dj].shape))
        z[di, dj] = avg + rval

        # In the square step, the pixel at the midpoint of each diamond is set to the
        # average value of the diamond's four corner pixels plus a random offset. This
        # step is a bit more complicated since (A) the pixels of interest form sort of a
        # checkerboard pattern rather than a regular grid, and (B) points located on the
        # border have only three neighboring pixels, not four. (A) is resolved by
        # splitting this update into two separate steps, each assigning values to a
        # different set of pixels. We address (B) by letting the indices wrap around so
        # a fourth neighbor is chosen from the opposing side of the array.

        # For the first set of diamonds, compute the average of the four corner pixels.
        ni, nj = uli, np.append(ulj, [[size - 1]], axis=1)
        si, sj = lli, np.append(llj, [[size - 1]], axis=1)
        ei, ej = di, np.append([[-start]], dj, axis=1)
        wi, wj = di, np.append(dj, [[-start]], axis=1)
        avg = 0.25 * (z[ni, nj] + z[si, sj] + z[ei, ej] + z[wi, wj])

        # Set the midpoint pixel to the average of the four corner pixels plus some
        # random value.
        si, sj = np.ogrid[start:size:stride, 0:size:stride]
        rval = scale * rng.uniform(-0.5, 0.5, size=(z[si, sj].shape))
        z[si, sj] = avg + rval

        # For the second set of diamonds, compute the average of the four corner pixels.
        ni, nj = np.append([[-start]], di, axis=0), dj
        si, sj = np.append(di, [[-start]], axis=0), dj
        ei, ej = np.append(uli, [[size - 1]], axis=0), ulj
        wi, wj = np.append(uri, [[size - 1]], axis=0), urj
        avg = 0.25 * (z[ni, nj] + z[si, sj] + z[ei, ej] + z[wi, wj])

        # Set the midpoint pixel to the average of the four corner pixels plus some
        # random value.
        si, sj = np.ogrid[0:size:stride, start:size:stride]
        rval = scale * rng.uniform(-0.5, 0.5, size=(z[si, sj].shape))
        z[si, sj] = avg + rval

        # At each iteration, the magnitude of the random value is reduced and the stride
        # is halved.
        scale *= 0.5**smoothness
        stride //= 2

    # Crop the output array to the desired dimensions.
    return z[:length, :width]
