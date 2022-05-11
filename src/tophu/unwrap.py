import dataclasses
from typing import Literal, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from isce3.io.gdal import Raster
from isce3.unwrap import snaphu

__all__ = [
    "SnaphuUnwrapper",
    "UnwrapFunc",
]


ComplexArray = NDArray[np.complexfloating]
FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]
UIntArray = NDArray[np.unsignedinteger]


@runtime_checkable
class UnwrapFunc(Protocol):
    """Abstract interface for phase unwrapping algorithms"""

    def __call__(
        self,
        igram: ComplexArray,
        corrcoef: FloatArray,
        nlooks: float,
        mask: Optional[BoolArray],
        unwest: Optional[FloatArray],
    ) -> Tuple[FloatArray, UIntArray]:
        """Performs two-dimensional phase unwrapping.

        Parameters
        ----------
        igram : numpy.ndarray
            Input interferogram.
        corrcoef : numpy.ndarray
            Sample correlation coefficient, normalized to the interval [0, 1], with the
            same shape as the input interferogram.
        nlooks : float
            Effective number of spatial looks used to form the input correlation data.
        mask : numpy.ndarray or None, optional
            Boolean mask of valid pixels, with the same shape as the input
            interferogram. A `True` value indicates that the corresponding pixel was
            valid. (default: None)
        unwest : numpy.ndarray or None, optional
            Initial estimate of the unwrapped phase, in radians, with the same shape as
            the input interferogram. (default: None)

        Returns
        -------
        unwphase : numpy.ndarray
            Unwrapped phase, in radians.
        conncomp : numpy.ndarray
            Connected component labels, with the same shape as the unwrapped phase.
        """
        ...


SnaphuCostMode = Literal["defo", "smooth", "p-norm"]
SnaphuCostParams = snaphu.CostParams
SnaphuInitMethod = Literal["mst", "mcf"]


@dataclasses.dataclass
class SnaphuUnwrapper(UnwrapFunc):
    """Performs two-dimensional phase unwrapping using SNAPHU [1]_.

    Attributes
    ----------
    cost : {'defo', 'smooth', 'p-norm'}
        Statistical cost mode.
    cost_params : isce3.unwrap.snaphu.CostParams or None
        Configuration parameters for the specified cost mode.
    init_method : {'mst', 'mcf'}
        Algorithm used for initialization of unwrapped phase gradients.

    References
    ----------
    .. [1] C. W. Chen and H. A. Zebker, "Two-dimensional phase unwrapping with use of
        statistical models for cost functions in nonlinear optimization," Journal of the
        Optical Society of America A, vol. 18, pp. 338-351 (2001).
    """

    cost: SnaphuCostMode = "smooth"
    cost_params: Optional[SnaphuCostParams] = None
    init_method: SnaphuInitMethod = "mcf"

    def __call__(
        self,
        igram: ComplexArray,
        corrcoef: FloatArray,
        nlooks: float,
        mask: Optional[BoolArray],
        unwest: Optional[FloatArray],
    ) -> Tuple[FloatArray, UIntArray]:

        # Convert input arrays to GDAL rasters.
        igram = Raster(igram)
        corrcoef = Raster(corrcoef)
        if mask is not None:
            mask = Raster(mask)
        if unwest is not None:
            unwest = Raster(unwest)

        # Get interferogram shape.
        shape = (igram.length, igram.width)

        # Create output arrays for unwrapped phase & connected component labels.
        unwphase = np.zeros(shape, dtype=np.float32)
        conncomp = np.zeros(shape, dtype=np.uint32)

        # Run SNAPHU.
        snaphu.unwrap(
            unw=Raster(unwphase),
            conncomp=Raster(conncomp),
            igram=igram,
            corr=corrcoef,
            nlooks=nlooks,
            cost=self.cost,
            cost_params=self.cost_params,
            init_method=self.init_method,
            mask=mask,
            unwest=unwest,
        )

        return unwphase, conncomp
