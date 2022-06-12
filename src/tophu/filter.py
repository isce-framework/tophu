import numpy as np
import scipy.signal
from numpy.typing import NDArray

__all__ = [
    "bandpass_equiripple_filter",
]


def equiripple_filter_order_kaiser(ptol: float, stol: float, width: float) -> int:
    r"""Estimate the order of an optimal equiripple FIR filter using Kaiser's formula.

    Predicts the order of a finite impulse response (FIR) digital filter obtained from
    the optimal equiripple approximation method using Kaiser's formula [1]_.

    Evaluates

    .. math::

        N = \left\lceil \frac{-10\log_{10}\left(\delta_p\delta_s\right)-13}
                {7.3\Delta f} \right\rceil

    where :math:`\lceil \cdot \rceil` is the ceil function, :math:`\delta_p` and
    :math:`\delta_s` are the peak ripple amplitudes in the passband and stopband,
    respectively, expressed in linear units, and :math:`\Delta f` is the normalized
    width of the transition band (the ratio of the transition bandwidth to the Nyquist
    rate).

    The filter length (or number of taps in the filter) is :math:`N + 1`.

    Parameters
    ----------
    ptol, stol : float
        Passband and stopband peak ripple amplitude, in linear units.
    width : float
        Width of the transition band, as a fraction of the Nyquist rate.

    Returns
    -------
    order : int
        Filter order.

    References
    ----------
    .. [1] Kaiser, James F. "Nonrecursive Digital Filter Design Using the I0-Sinh Window
        Function." Proceedings of the 1974 IEEE International Symposium on Circuits and
        Systems. 1974, pp. 20-23.
    """
    assert ptol > 0.0
    assert stol > 0.0
    assert (width > 0.0) and (width < 1.0)

    return int(np.ceil((-10.0 * np.log10(ptol * stol) - 13.0) / (7.3 * width)))


def bandpass_equiripple_filter(
    bandwidth: float,
    shape: float,
    ripple: float,
    attenuation: float,
    centerfreq: float = 0.0,
    samplerate: float = 1.0,
    *,
    force_odd_length: bool = False,
    maxiter: int = 25,
    grid_density: int = 16,
) -> NDArray:
    """Design a bandpass FIR digital filter using the Parks-McClellan algorithm.

    Form a linear-phase finite impulse response (FIR) discrete-time digital filter wth
    desired passband, stopband, and attenuation characteristics using the
    Parks-McClellan algorithm [1]_. The algorithm produces a filter is optimal in the
    sense that it minimizes the maximum weighted deviation from the ideal frequency
    response in the passband and stopband.

    The resulting filter has equiripple frequency response in both the passband and
    stopband.

    Parameters
    ----------
    bandwidth : float
        Width of the passband, in the same units as `samplerate`.
    shape : float
        Shape factor -- the ratio of the width of the combined passband and transition
        band to the passband width.
    ripple : float
        Passband ripple -- the maximum allowed ripple amplitude below unity gain in the
        passband, in decibels.
    attenuation : float
        Stopband attenuation -- the difference in amplitude, in decibels, between the
        ideal gain in the passband and the highest gain in the stopband.
    centerfreq : float, optional
        Center frequency of the passband, in the same units as `samplerate`.
        (default: 0)
    samplerate : float, optional
        Sampling frequency of the signal. (default: 1)
    force_odd_length : bool, optional
        Whether to force the filter length to be odd-valued. (default: False)
    maxiter : int, optional
        Maximum number of iterations of the algorithm. (default: 25)
    grid_density : int, optional
        Density of Lagrange interpolation points used in the algorithm. (default: 16)

    Returns
    -------
    coeffs : numpy.ndarray
        Filter coefficients.

    References
    ----------
    .. [1] J. H. McClellan and T. W. Parks, "A unified approach to the design of optimum
        FIR linear phase digital filters", IEEE Trans. Circuit Theory, vol. CT-20, pp.
        697-701, 1973.
    """
    if (bandwidth <= 0.0) or (bandwidth >= samplerate):
        raise ValueError(f"passband width must be > 0 and < {samplerate}")
    max_shape = samplerate / bandwidth
    if (shape <= 1.0) or (shape >= max_shape):
        raise ValueError(f"shape factor must be > 1 and < {max_shape}")
    if ripple == 0.0:
        raise ValueError("passband ripple must be nonzero")
    if attenuation == 0.0:
        raise ValueError("stopband attenuation must be nonzero")
    if samplerate <= 0.0:
        raise ValueError("sample rate must be > 0")
    if maxiter <= 1:
        raise ValueError("max number of iterations must be >= 1")
    if grid_density <= 1:
        raise ValueError("grid density must be >= 1")

    # Below, we expect `ripple` and `attenuation` to have positive values, but it's also
    # valid to interpret them as negative quantities. We should correctly handle either
    # definition.
    ripple = abs(ripple)
    attenuation = abs(attenuation)

    # Passband & stopband cutoff frequencies.
    wp = 0.5 * bandwidth
    ws = 0.5 * shape * bandwidth

    # Nyquist rate.
    nyq = 0.5 * samplerate

    # Transition band width, as a fraction of the Nyquist rate.
    dw = (ws - wp) / nyq

    # Converts a field quantity from decibels to linear units.
    def db2amp(x: float) -> float:
        return 10.0 ** (0.05 * x)

    # Passband & stopband ripple amplitude tolerances.
    dp = 1.0 - db2amp(-ripple)
    ds = db2amp(-attenuation)

    # Estimate filter order using Kaiser's formula.
    N = equiripple_filter_order_kaiser(dp, ds, dw)

    # Checks if an integer is odd-valued.
    def isodd(n: int) -> bool:
        return n & 1 == 1

    # If `force_odd_length` was specified, force the filter order to by even-valued
    # (thereby forcing the filter *length* to be odd-valued).
    if force_odd_length and isodd(N):
        N += 1

    # Filter length.
    numtaps = N + 1

    # Edge frequencies of each desired band. The first two elements are the start & end
    # points of the passband frequency interval; the last two elements are the bounds of
    # stopband frequency interval.
    edgefreqs = [0.0, wp, ws, nyq]

    # Desired gain for each band (unity in the passband and zero in the stopband).
    gains = [1.0, 0.0]

    # Relative weights for each band.
    weights = [ds, dp]

    # Get filter coefficients.
    coeffs = scipy.signal.remez(
        numtaps,
        edgefreqs,
        gains,
        weights,
        maxiter=maxiter,
        grid_density=grid_density,
        fs=samplerate,
    )

    # Shifts the frequency of a discrete-time signal, sampled at rate `fs`, by frequency
    # `f`.
    def freqshift(signal, f, fs):
        n = len(signal)
        dt = 1.0 / fs
        t = dt * np.arange(n)
        phase = 2.0 * np.pi * f * t
        return signal * np.exp(1.0j * phase)

    # If `centerfreq` is nonzero, convert lowpass filter to bandpass filter by up/down
    # conversion (frequency shift).
    if centerfreq != 0.0:
        coeffs = freqshift(coeffs, centerfreq, samplerate)

    return coeffs
