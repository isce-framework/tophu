from __future__ import annotations

import dataclasses
import warnings
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Protocol, runtime_checkable

import isce3
import numpy as np
import rasterio
from numpy.typing import DTypeLike, NDArray
from rasterio.errors import NotGeoreferencedWarning

__all__ = [
    "ICUUnwrap",
    "PhassUnwrap",
    "SnaphuUnwrap",
    "UnwrapCallback",
]


@runtime_checkable
class UnwrapCallback(Protocol):
    """
    Callback protocol for two-dimensional phase unwrapping algorithms.

    `UnwrapCallback` defines the abstract interface that unwrapping implementations are
    expected to conform to in order to "plug in" to the multi-scale unwrapping
    framework.
    """

    def __call__(
        self,
        igram: NDArray[np.complexfloating],
        coherence: NDArray[np.floating],
        nlooks: float,
    ) -> tuple[NDArray[np.floating], NDArray[np.unsignedinteger]]:
        """
        Perform two-dimensional phase unwrapping.

        Parameters
        ----------
        igram : numpy.ndarray
            Input interferogram.
        coherence : numpy.ndarray
            Sample coherence coefficient, normalized to the interval [0, 1], with the
            same shape as the input interferogram.
        nlooks : float
            Effective number of spatial looks used to form the input coherence data.

        Returns
        -------
        unwphase : numpy.ndarray
            Unwrapped phase, in radians.
        conncomp : numpy.ndarray
            Connected component labels, with the same shape as the unwrapped phase.
        """
        ...


@dataclasses.dataclass
class SnaphuUnwrap(UnwrapCallback):
    r"""
    Callback functor for unwrapping using SNAPHU.

    Performs unwrapping using the SNAPHU algorithm\ :footcite:p:`chen2001`.
    """

    cost: str
    """str : Statistical cost mode."""

    cost_params: isce3.unwrap.snaphu.CostParams | None
    """
    isce3.unwrap.snaphu.CostParams or None : Configuration parameters for the
    specified cost mode.
    """

    init_method: str
    """str : Initialization method."""

    def __init__(
        self,
        cost: Literal["topo", "defo", "smooth", "p-norm"] = "smooth",
        cost_params: isce3.unwrap.snaphu.CostParams | None = None,
        init_method: Literal["mst", "mcf"] = "mcf",
    ):
        """
        Construct a new `SnaphuUnwrap` object.

        Parameters
        ----------
        cost : {'topo', 'defo', 'smooth', 'p-norm'}, optional
            Statistical cost mode. Defaults to 'smooth'.
        cost_params : isce3.unwrap.snaphu.CostParams or None, optional
            Configuration parameters for the specified cost mode. Defaults to None.
        init_method : {'mst', 'mcf'}, optional
            Initialization method. Defaults to 'mcf'.
        """
        if cost not in {"topo", "defo", "smooth", "p-norm"}:
            raise ValueError(f"unexpected cost mode '{cost}'")
        if init_method not in {"mst", "mcf"}:
            raise ValueError(f"unexpected initialization method '{init_method}'")

        self.cost = cost
        self.cost_params = cost_params
        self.init_method = init_method

    def __call__(
        self,
        igram: NDArray[np.complexfloating],
        coherence: NDArray[np.floating],
        nlooks: float,
    ) -> tuple[NDArray[np.floating], NDArray[np.unsignedinteger]]:
        # Convert input arrays to GDAL rasters with the expected datatypes.
        igram_data = np.asanyarray(igram, dtype=np.complex64)
        igram = isce3.io.gdal.Raster(igram_data)

        coherence_data = np.asanyarray(coherence, dtype=np.float32)
        coherence = isce3.io.gdal.Raster(coherence_data)

        # Get interferogram shape.
        shape = (igram.length, igram.width)

        # Create output arrays for unwrapped phase & connected component labels.
        unwphase = np.zeros(shape, dtype=np.float32)
        conncomp = np.zeros(shape, dtype=np.uint32)

        # Run SNAPHU.
        isce3.unwrap.snaphu.unwrap(
            unw=isce3.io.gdal.Raster(unwphase),
            conncomp=isce3.io.gdal.Raster(conncomp),
            igram=igram,
            corr=coherence,
            nlooks=nlooks,
            cost=self.cost,
            cost_params=self.cost_params,
            init_method=self.init_method,
        )

        return unwphase, conncomp


def isodd(n: int) -> bool:
    """Check if the input is odd-valued."""
    return n & 1 == 1


def create_geotiff(
    path: PathLike,
    *,
    width: int,
    length: int,
    dtype: DTypeLike,
) -> isce3.io.Raster:
    """
    Create a new single-band GeoTiff dataset.

    Parameters
    ----------
    path : path_like
        Filesystem path of the new dataset.
    width : int
        Raster width.
    length : int
        Raster length.
    dtype : dtype_like
        Raster datatype.

    Returns
    -------
    dataset : isce3.io.Raster
        Created dataset, opened in read/write mode.
    """
    # Create a new dataset.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NotGeoreferencedWarning)
        with rasterio.open(
            path,
            mode="w",
            driver="GTiff",
            width=width,
            height=length,
            count=1,
            dtype=dtype,
        ):
            pass

    # Open the dataset as an `isce3.io.Raster` in read/write mode.
    return isce3.io.Raster(str(path), update=True)


def to_geotiff(path: PathLike, arr: NDArray) -> isce3.io.Raster:
    """
    Write array data to a new single-band GeoTiff dataset.

    Parameters
    ----------
    path : path_like
        Filesystem path of the new dataset.
    arr : numpy.ndarray
        2-dimensional array.

    Returns
    -------
    dataset : isce3.io.Raster
        Created dataset, opened in read-only mode.
    """
    # Create a new dataset.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NotGeoreferencedWarning)
        with rasterio.open(
            path,
            mode="w",
            driver="GTiff",
            width=arr.shape[1],
            height=arr.shape[0],
            count=1,
            dtype=arr.dtype,
        ) as dataset:
            # Write array data to first band in dataset.
            dataset.write(arr, 1)

    # Open the dataset as an `isce3.io.Raster` in read-only mode.
    return isce3.io.Raster(str(path))


def read_raster(path: PathLike, band: int = 1) -> NDArray:
    """
    Read raster data from a dataset as an array.

    Parameters
    ----------
    path : path_like
        Filesystem path of the dataset to read.
    band : int, optional
        (1-based) band index of the raster to read. Defaults to 1.

    Returns
    -------
    arr : numpy.ndarray
        Array containing raster data.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NotGeoreferencedWarning)
        with rasterio.open(path, mode="r") as dataset:
            return dataset.read(band)


@dataclasses.dataclass
class ICUUnwrap(UnwrapCallback):
    """Callback functor for unwrapping using ICU."""

    min_coherence: float
    """
    float : Minimum coherence of valid data.

    Pixels with lower coherence are masked out.
    """

    ntrees: int
    """int : Number of randomized tree realizations to generate."""

    max_branch_length: int
    """int : Maximum branch cut length, in pixels."""

    use_phasegrad_neutrons: bool
    """bool : Whether to use phase gradient neutrons."""

    use_intensity_neutrons: bool
    """bool : Whether to use intensity neutrons."""

    phasegrad_window_size: int
    """
    int : Window size for estimating phase gradients.

    This parameter is ignored if `use_phasegrad_neutrons` is false.
    """

    neutron_phasegrad_thresh: float
    """
    float : Neutron absolute phase gradient threshold.

    Absolute phase gradient threshold for detecting phase gradient neutrons, in radians
    per sample.

    This parameter is ignored if `use_phasegrad_neutrons` is false.
    """

    neutron_intensity_thresh: float
    """
    float : Neutron intensity standard deviation threshold.

    Intensity variation threshold for detecting intensity neutrons, in standard
    deviations from the mean (based on local image statistics).

    This parameter is ignored if `use_intensity_neutrons` is false.
    """

    neutron_coherence_thresh: float
    """
    float : Coherence threshold for detecting intensity neutrons.

    This parameter is ignored if `use_intensity_neutrons` is false.
    """

    min_conncomp_area_frac: float
    """
    float : Minimum connected component size fraction.

    Minimum connected component area as a fraction of the total size of the
    interferogram tile.
    """

    def __init__(
        self,
        min_coherence: float = 0.1,
        ntrees: int = 7,
        max_branch_length: int = 64,
        use_phasegrad_neutrons: bool = False,
        use_intensity_neutrons: bool = False,
        phasegrad_window_size: int = 5,
        neutron_phasegrad_thresh: float = 3.0,
        neutron_intensity_thresh: float = 8.0,
        neutron_coherence_thresh: float = 0.8,
        min_conncomp_area_frac: float = 1.0 / 320.0,
    ):
        """
        Construct a new `ICUUnwrap` object.

        Parameters
        ----------
        min_coherence : float, optional
            Minimum coherence of valid data. Pixels with lower coherence are masked out.
            Defaults to 0.1.
        ntrees : int, optional
            Number of randomized tree realizations to generate. Defaults to 7.
        max_branch_length : int, optional
            Maximum length of a branch cut between residues/neutrons, in pixels.
            Defaults to 64.
        use_phasegrad_neutrons : bool, optional
            Whether to use phase gradient neutrons. Defaults to False.
        use_intensity_neutrons : bool, optional
            Whether to use intensity neutrons. Defaults to False.
        phasegrad_window_size : int, optional
            Window size for estimating phase gradients. This parameter is ignored if
            `use_phasegrad_neutrons` was false. Defaults to 5.
        neutron_phasegrad_thresh : float, optional
            Absolute phase gradient threshold for detecting phase gradient neutrons, in
            radians per sample. This parameter is ignored if `use_phasegrad_neutrons`
            was false. Defaults to 3.
        neutron_intensity_thresh : float, optional
            Intensity variation threshold for detecting intensity neutrons, in standard
            deviations from the mean (based on local image statistics). This parameter
            is ignored if `use_intensity_neutrons` was false. Defaults to 8.
        neutron_coherence_thresh : float, optional
            Sample coherence threshold for detecting intensity neutrons. This parameter
            is ignored if `use_intensity_neutrons` was false. Defaults to 0.8.
        min_conncomp_area_frac : float, optional
            Minimum connected component size as a fraction of the total size of the
            interferogram tile. Defaults to 1/320.
        """
        if not (0.0 <= min_coherence <= 1.0):
            raise ValueError("minimum coherence must be between 0 and 1")
        if ntrees < 1:
            raise ValueError("number of tree realizations must be >= 1")
        if max_branch_length < 1:
            raise ValueError("max branch length must be >= 1")
        if phasegrad_window_size < 1:
            raise ValueError("phase gradient window size must be >= 1")
        if not isodd(phasegrad_window_size):
            raise ValueError("phase gradient window size must be odd-valued")
        if neutron_phasegrad_thresh <= 0.0:
            raise ValueError("neutron phase gradient threshold must be > 0")
        if neutron_intensity_thresh <= 0.0:
            raise ValueError("neutron intensity variation threshold must be > 0")
        if not (0.0 <= neutron_coherence_thresh <= 1.0):
            raise ValueError("neutron coherence threshold must be between 0 and 1")
        if min_conncomp_area_frac <= 0.0:
            raise ValueError("minimum connected component size must be > 0")

        self.min_coherence = min_coherence
        self.ntrees = ntrees
        self.max_branch_length = max_branch_length
        self.use_phasegrad_neutrons = use_phasegrad_neutrons
        self.use_intensity_neutrons = use_intensity_neutrons
        self.phasegrad_window_size = phasegrad_window_size
        self.neutron_phasegrad_thresh = neutron_phasegrad_thresh
        self.neutron_intensity_thresh = neutron_intensity_thresh
        self.neutron_coherence_thresh = neutron_coherence_thresh
        self.min_conncomp_area_frac = min_conncomp_area_frac

    def __call__(
        self,
        igram: NDArray[np.complexfloating],
        coherence: NDArray[np.floating],
        nlooks: float,
    ) -> tuple[NDArray[np.floating], NDArray[np.unsignedinteger]]:
        # Configure ICU to unwrap the interferogram as a single tile (no bootstrapping).
        icu = isce3.unwrap.ICU(
            buffer_lines=len(igram),
            use_phase_grad_neut=self.use_phasegrad_neutrons,
            use_intensity_neut=self.use_intensity_neutrons,
            phase_grad_win_size=self.phasegrad_window_size,
            neut_phase_grad_thr=self.neutron_phasegrad_thresh,
            neut_intensity_thr=self.neutron_intensity_thresh,
            neut_correlation_thr=self.neutron_coherence_thresh,
            trees_number=self.ntrees,
            max_branch_length=self.max_branch_length,
            init_corr_thr=self.min_coherence,
            min_cc_area=self.min_conncomp_area_frac,
        )

        # Create a temporary scratch directory to store intermediate rasters.
        with TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)

            # Convert input arrays into rasters.
            igram_raster = to_geotiff(d / "igram.tif", igram)
            coherence_raster = to_geotiff(d / "coherence.tif", coherence)

            # Create zero-filled rasters to store output data.
            unwphase_raster = create_geotiff(
                d / "unwphase.tif",
                width=igram.shape[1],
                length=igram.shape[0],
                dtype=np.float32,
            )
            conncomp_raster = create_geotiff(
                d / "conncomp.tif",
                width=igram.shape[1],
                length=igram.shape[0],
                dtype=np.uint8,
            )

            # Run ICU.
            icu.unwrap(unwphase_raster, conncomp_raster, igram_raster, coherence_raster)

            # Ensure changes to output rasters are flushed to disk and close the files.
            del unwphase_raster
            del conncomp_raster

            # Read output rasters into in-memory arrays.
            unwphase = read_raster(d / "unwphase.tif")
            conncomp = read_raster(d / "conncomp.tif")

        return unwphase, conncomp


@dataclasses.dataclass
class PhassUnwrap(UnwrapCallback):
    """Callback functor for unwrapping using PHASS."""

    coherence_thresh: float
    good_coherence: float
    min_region_size: int

    def __init__(
        self,
        coherence_thresh: float = 0.2,
        good_coherence: float = 0.7,
        min_region_size: int = 200,
    ):
        """
        Construct a new `PhassUnwrap` object.

        Parameters
        ----------
        coherence_thresh : float, optional
            ??? Defaults to 0.2.
        good_coherence : float, optional
            ??? Defaults to 0.7.
        min_region_size : int, optional
            ??? Defaults to 200.
        """
        if not (0.0 <= coherence_thresh <= 1.0):
            raise ValueError("coherence threshold must be between 0 and 1")
        if not (0.0 <= good_coherence <= 1.0):
            raise ValueError("good coherence must be between 0 and 1")
        if min_region_size <= 0:
            raise ValueError("minimum region size must be > 0")

        self.coherence_thresh = coherence_thresh
        self.good_coherence = good_coherence
        self.min_region_size = min_region_size

    def __call__(
        self,
        igram: NDArray[np.complexfloating],
        coherence: NDArray[np.floating],
        nlooks: float,
    ) -> tuple[NDArray[np.floating], NDArray[np.unsignedinteger]]:
        # Configure ICU to unwrap the interferogram as a single tile (no bootstrapping).
        phass = isce3.unwrap.Phass(
            correlation_threshold=self.coherence_thresh,
            good_correlation=self.good_coherence,
            min_pixels_region=self.min_region_size,
        )

        # Get wrapped phase.
        wphase = np.angle(igram)

        # Create a temporary scratch directory to store intermediate rasters.
        with TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)

            # Convert input arrays into rasters.
            wphase_raster = to_geotiff(d / "wphase.tif", wphase)
            coherence_raster = to_geotiff(d / "coherence.tif", coherence)

            # Create zero-filled rasters to store output data.
            unwphase_raster = create_geotiff(
                d / "unwphase.tif",
                width=igram.shape[1],
                length=igram.shape[0],
                dtype=np.float32,
            )
            conncomp_raster = create_geotiff(
                d / "conncomp.tif",
                width=igram.shape[1],
                length=igram.shape[0],
                dtype=np.uint32,
            )

            # Run PHASS.
            phass.unwrap(
                wphase_raster,
                coherence_raster,
                unwphase_raster,
                conncomp_raster,
            )

            # Ensure changes to output rasters are flushed to disk and close the files.
            del unwphase_raster
            del conncomp_raster

            # Read output rasters into in-memory arrays.
            unwphase = read_raster(d / "unwphase.tif")
            conncomp = read_raster(d / "conncomp.tif")

        return unwphase, conncomp
