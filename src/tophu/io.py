import mmap
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Tuple, Union, overload, runtime_checkable

import h5py
import numpy as np
import rasterio
from numpy.typing import ArrayLike, DTypeLike

from . import util

__all__ = [
    "DatasetReader",
    "DatasetWriter",
    "BinaryFile",
    "HDF5Dataset",
    "RasterBand",
]


@runtime_checkable
class DatasetReader(Protocol):
    """
    An array-like interface for reading input datasets.

    `DatasetReader` defines the abstract interface that types must conform to in order
    to be valid inputs to the `multiscale_unwrap()` function. Such objects must export
    NumPy-like `dtype`, `shape`, and `ndim` attributes and must support NumPy-style
    slice-based indexing.

    See Also
    --------
    DatasetWriter
    """

    dtype: np.dtype
    """numpy.dtype : Data-type of the array's elements."""

    shape: Tuple[int, ...]
    """tuple of int : Tuple of array dimensions."""

    ndim: int
    """int : Number of array dimensions."""

    def __getitem__(self, key: Tuple[slice, ...], /) -> ArrayLike:
        """Read a block of data."""
        ...


@runtime_checkable
class DatasetWriter(Protocol):
    """
    An array-like interface for writing output datasets.

    `DatasetWriter` defines the abstract interface that types must conform to in order
    to be valid outputs of the `multiscale_unwrap()` function. Such objects must export
    NumPy-like `dtype`, `shape`, and `ndim` attributes and must support NumPy-style
    slice-based indexing.

    See Also
    --------
    DatasetReader
    """

    dtype: np.dtype
    """numpy.dtype : Data-type of the array's elements."""

    shape: Tuple[int, ...]
    """tuple of int : Tuple of array dimensions."""

    ndim: int
    """int : Number of array dimensions."""

    def __setitem__(self, key: Tuple[slice, ...], value: np.ndarray, /) -> None:
        """Write a block of data."""
        ...


def _create_or_extend_file(filepath: Union[str, os.PathLike[str]], size: int) -> None:
    """
    Create a file with the specified size or extend an existing file to the same size.

    Parameters
    ----------
    filepath : str or path-like
        File path.
    size : int
        The size, in bytes, of the file.
    """
    filepath = Path(filepath)

    if not filepath.is_file():
        # If the file does not exist, then create it with the specified size.
        with filepath.open("wb") as f:
            f.truncate(size)
    else:
        # If the file exists but is smaller than the requested size, extend the file
        # length.
        filesize = filepath.stat().st_size
        if filesize < size:
            with filepath.open("r+b") as f:
                f.truncate(size)


@dataclass(frozen=True)
class BinaryFile(DatasetReader, DatasetWriter):
    """
    A raw binary file for storing array data.

    See Also
    --------
    HDF5Dataset
    RasterBand

    Notes
    -----
    This class does not store an open file object. Instead, the file is opened on-demand
    for reading or writing and closed immediately after each read/write operation. This
    allows multiple spawned processes to write to the file in coordination (as long as a
    suitable mutex is used to guard file access.)
    """

    filepath: Path
    """pathlib.Path : The file path."""

    shape: Tuple[int, ...]
    dtype: np.dtype

    def __init__(
        self,
        filepath: Union[str, os.PathLike[str]],
        shape: Tuple[int, ...],
        dtype: DTypeLike,
    ):
        """
        Construct a new `BinaryFile` object.

        If the file does not exist, it will be created. If the file does exist but is
        smaller than the array, it will be extended to the size (in bytes) of the array.

        Parameters
        ----------
        filepath : str or path-like
            The file path.
        shape : tuple of int
            Tuple of array dimensions.
        dtype : data-type
            Data-type of the array's elements. Must be convertible to a `numpy.dtype`
            object.
        """
        filepath = Path(filepath)
        shape = util.as_tuple_of_int(shape)
        dtype = np.dtype(dtype)

        # Get array size in bytes.
        length = np.prod(shape) * dtype.itemsize

        # If the file doesn't exist, create it with the required size. Else, ensure that
        # the file size is at least `length` bytes.
        _create_or_extend_file(filepath, length)

        # Workaround for `frozen=True`.
        object.__setattr__(self, "filepath", filepath)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype)

    @property
    def ndim(self) -> int:  # type: ignore[override]
        """int : Number of array dimensions."""
        return len(self.shape)

    def __array__(self) -> np.ndarray:
        return self[:,]

    def __getitem__(self, key: Tuple[slice, ...], /) -> np.ndarray:
        with self.filepath.open("rb") as f:
            # Memory-map the entire file.
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                # In order to safely close the memory-map, there can't be any dangling
                # references to it, so we return a copy (not a view) of the requested
                # data and decref the array object.
                arr = np.frombuffer(mm, dtype=self.dtype).reshape(self.shape)
                data = arr[key].copy()
                del arr
            return data

    def __setitem__(self, key: Tuple[slice, ...], value: np.ndarray, /) -> None:
        with self.filepath.open("r+b") as f:
            # Memory-map the entire file.
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE) as mm:
                # In order to safely close the memory-map, there can't be any dangling
                # references to it, so we decref the array object after writing the
                # data.
                arr = np.frombuffer(mm, dtype=self.dtype).reshape(self.shape)
                arr[key] = value
                mm.flush()
                del arr


@dataclass(frozen=True)
class HDF5Dataset(DatasetReader, DatasetWriter):
    """
    A Dataset in an HDF5 file.

    See Also
    --------
    BinaryFile
    RasterBand

    Notes
    -----
    This class does not store an open file object. Instead, the file is opened on-demand
    for reading or writing and closed immediately after each read/write operation. This
    allows multiple spawned processes to write to the file in coordination (as long as a
    suitable mutex is used to guard file access.)
    """

    filepath: Path
    """pathlib.Path : The file path."""

    datapath: str
    """str : The path to the dataset within the file."""

    chunks: Optional[Tuple[int, ...]]
    """
    tuple of int : Tuple giving the chunk shape, or None if chunked storage is not used.
    """

    shape: Tuple[int, ...]
    dtype: np.dtype

    @overload
    def __init__(
        self, filepath: Union[str, os.PathLike[str]], datapath: str
    ):  # noqa: D418
        """
        Construct a new `HDF5Dataset` object from an existing dataset.

        Parameters
        ----------
        filepath : str or path-like
            The file path.
        datapath : str
            The path to the dataset within the file.
        """
        ...

    @overload
    def __init__(
        self,
        filepath: Union[str, os.PathLike[str]],
        datapath: str,
        shape: Tuple[int, ...],
        dtype: DTypeLike,
    ):  # noqa: D418
        """
        Construct a new `HDF5Dataset` object.

        The HDF5 file and dataset will be created if either do not exist.

        Parameters
        ----------
        filepath : str or path-like
            The file path.
        datapath : str
            The path to the dataset within the file.
        shape : tuple of int
            Tuple of array dimensions.
        dtype : data-type
            Data-type of the array's elements. Must be convertible to a `numpy.dtype`
            object.
        kwargs : dict
            Additional dataset creation options. These keywords are ignored if a dataset
            is not created. See `h5py.Group.create_dataset()` for valid options.
        """
        ...

    def __init__(
        self, filepath, datapath, shape=None, dtype=None, **kwargs
    ):  # noqa: D107
        filepath = Path(filepath)

        # If either `shape` or `dtype` is provided, both must be provided. If they
        # weren't specified, the dataset must already exist.
        if (shape is None) and (dtype is None):
            # Open the dataset and get its shape, dtype, and chunk shape (if
            # applicable).
            with h5py.File(filepath, "r") as f:
                dataset = f[datapath]
                shape = dataset.shape
                dtype = dataset.dtype
                chunks = dataset.chunks
        elif (shape is not None) and (dtype is not None):
            # Create the HDF5 file and dataset if they don't already exist.
            # If the dataset already exists, make sure its shape & dtype are as
            # specified.
            shape = util.as_tuple_of_int(shape)
            dtype = np.dtype(dtype)
            with h5py.File(filepath, "a") as f:
                dataset = f.require_dataset(
                    datapath,
                    shape=shape,
                    dtype=dtype,
                    exact=True,
                    **kwargs,
                )
                chunks = dataset.chunks
        else:
            errmsg = (
                "the supplied arguments don't match any valid overload of HDF5Dataset"
            )
            raise TypeError(errmsg)

        # Workaround for `frozen=True`.
        object.__setattr__(self, "filepath", filepath)
        object.__setattr__(self, "datapath", datapath)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "chunks", chunks)

    @property
    def ndim(self) -> int:  # type: ignore[override]
        """int : Number of array dimensions."""
        return len(self.shape)

    def __array__(self) -> np.ndarray:
        return self[:,]

    def __getitem__(self, key: Tuple[slice, ...], /) -> np.ndarray:
        with h5py.File(self.filepath, "r") as f:
            dataset = f[self.datapath]
            return dataset[key]

    def __setitem__(self, key: Tuple[slice, ...], value: np.ndarray, /) -> None:
        with h5py.File(self.filepath, "r+") as f:
            dataset = f[self.datapath]
            dataset[key] = value


def _check_contains_single_band(
    dataset: Union[rasterio.io.DatasetReader, rasterio.io.DatasetWriter]
) -> None:
    """
    Validate that the supplied dataset contains a single raster band.

    Parameters
    ----------
    dataset : rasterio.io.DatasetReader or rasterio.io.DatasetWriter
        The raster dataset.

    Raises
    ------
    ValueError
        If `dataset` contained multiple bands.
    """
    nbands = dataset.count
    if nbands != 1:
        errmsg = f"dataset contains {nbands} raster bands: band index must be specified"
        raise ValueError(errmsg)


def _check_valid_band(
    dataset: Union[rasterio.io.DatasetReader, rasterio.io.DatasetWriter],
    band: int,
) -> None:
    """
    Ensure that the band index is valid.

    Parameters
    ----------
    dataset : rasterio.io.DatasetReader or rasterio.io.DatasetWriter
        The raster dataset.
    band : int
        The (1-based) band index of the raster band.

    Raises
    ------
    ValueError
        If the band index was out of range for the supplied dataset.
    """
    nbands = dataset.count
    if not (1 <= band <= nbands):
        errmsg = (
            f"band index {band} out of range: dataset contains {nbands} raster bands"
        )
        raise IndexError(errmsg)


@dataclass(frozen=True)
class RasterBand(DatasetReader, DatasetWriter):
    """
    A single raster band in a GDAL-compatible dataset containing one or more bands.

    See Also
    --------
    BinaryFile
    HDF5Dataset

    Notes
    -----
    This class does not store an open file object. Instead, the file is opened on-demand
    for reading or writing and closed immediately after each read/write operation. This
    allows multiple spawned processes to write to the file in coordination (as long as a
    suitable mutex is used to guard file access.)
    """

    filepath: Path
    """pathlib.Path : The file path."""

    band: int
    """int : Band index (1-based)."""

    driver: str
    """str : Raster format driver name."""

    crs: rasterio.crs.CRS
    """rasterio.crs.CRS : The dataset's coordinate reference system."""

    transform: rasterio.transform.Affine
    """
    rasterio.transform.Affine : The dataset's georeferencing transformation matrix.

    This transform maps pixel row/column coordinates to coordinates in the dataset's
    coordinate reference system.
    """

    shape: Tuple[int, int]
    dtype: np.dtype

    # TODO: `chunks` & `nodata` attributes

    @overload
    def __init__(
        self,
        filepath: Union[str, os.PathLike[str]],
        *,
        band: Optional[int] = None,
        driver: Optional[str] = None,
    ):  # noqa: D418
        """
        Construct a new `RasterBand` object.

        Parameters
        ----------
        filepath : str or path-like
            Path of the local or remote dataset.
        band : int or None, optional
            The (1-based) band index of the raster band. Must be specified if the
            dataset contains multiple bands. (default: None)
        driver : str or or None, optional
            Raster format driver name. If None, registered drivers will be tried
            sequentially until a match is found. (default: None)
        """
        ...

    @overload
    def __init__(
        self,
        filepath: Union[str, os.PathLike[str]],
        width: int,
        height: int,
        dtype: DTypeLike,
        *,
        driver: Optional[str] = None,
        crs: Optional[Union[str, dict, rasterio.crs.CRS]] = None,
        transform: Optional[rasterio.transform.Affine] = None,
    ):  # noqa: D418
        """
        Construct a new `RasterBand` object.

        Parameters
        ----------
        filepath : str or path-like
            A remote or local dataset path.
        width, height : int
            The numbers of columns and rows of the raster dataset.
        dtype : data-type
            Data-type of the raster dataset's elements. Must be convertible to a
            `numpy.dtype` object and must correspond to a valid GDAL datatype.
        driver : str or None, optional
            Raster format driver name. If None, the method will attempt to infer the
            driver from the file extension. (default: None)
        crs : str, dict, or CRS; optional
            The coordinate reference system. (default: None)
        transform : Affine instance, optional
            Affine transformation mapping the pixel space to geographic space.
            (default: None)
        """
        ...

    def __init__(
        self,
        filepath,
        width=None,
        height=None,
        dtype=None,
        *,
        band=None,
        driver=None,
        crs=None,
        transform=None,
    ):  # noqa: D107
        filepath = Path(filepath)

        # If any of `width`, `height`, or `dtype` are provided, all three must be
        # provided. If any were not provided, the dataset must already exist. Otherwise,
        # create the dataset if it did not exist.
        if (width is None) and (height is None) and (dtype is None):
            mode = "r"
            count = None
        elif (width is not None) and (height is not None) and (dtype is not None):
            mode = "w+"
            count = 1
        else:
            errmsg = (
                "the supplied arguments don't match any valid overload of RasterBand"
            )
            raise TypeError(errmsg)

        # Create the dataset if it didn't exist.
        with rasterio.open(
            filepath,
            mode,
            driver=driver,
            width=width,
            height=height,
            count=count,
            crs=crs,
            transform=transform,
            dtype=dtype,
        ) as dataset:
            # Band index must not be None if the dataset contains more than one band.
            # If a band index was supplied, check that it's within the range of valid
            # band indices.
            if band is None:
                _check_contains_single_band(dataset)
                band = 1
            else:
                _check_valid_band(dataset, band)

            shape = (dataset.height, dataset.width)
            dtype = np.dtype(dataset.dtypes[band - 1])
            driver = dataset.driver
            crs = dataset.crs
            transform = dataset.transform

        # Workaround for `frozen=True`.
        object.__setattr__(self, "filepath", filepath)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "band", band)
        object.__setattr__(self, "driver", driver)
        object.__setattr__(self, "crs", crs)
        object.__setattr__(self, "transform", transform)

    @property
    def ndim(self) -> int:  # type: ignore[override]
        """int : Number of array dimensions."""
        return 2

    def __array__(self) -> np.ndarray:
        return self[:, :]

    def __getitem__(self, key: Tuple[slice, ...], /) -> np.ndarray:
        with rasterio.io.DatasetReader(
            self.filepath,
            driver=self.driver,
        ) as dataset:
            window = rasterio.windows.Window.from_slices(
                *key,
                height=dataset.height,
                width=dataset.width,
            )
            return dataset.read(self.band, window=window)

    def __setitem__(self, key: Tuple[slice, ...], value: np.ndarray, /) -> None:
        with rasterio.io.DatasetWriter(
            self.filepath,
            "r+",
            driver=self.driver,
        ) as dataset:
            window = rasterio.windows.Window.from_slices(
                *key,
                height=dataset.height,
                width=dataset.width,
            )
            return dataset.write(value, self.band, window=window)
