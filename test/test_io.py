from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

import h5py
import numpy as np
import pytest
import rasterio
from numpy.typing import DTypeLike

import tophu


def filesize(filepath: str | os.PathLike) -> int:
    """Get file size in bytes."""
    return Path(filepath).stat().st_size


def valid_gdal_dtypes() -> tuple[np.dtype, ...]:
    return (
        np.uint8,
        np.int8,
        np.uint16,
        np.int16,
        np.uint32,
        np.int32,
        np.uint64,
        np.int64,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    )


def create_raster_dataset(
    filepath: str | os.PathLike,
    driver: str,
    shape: tuple[int, int],
    count: int,
    dtype: DTypeLike,
) -> None:
    """Create a new raster dataset."""
    with rasterio.open(
        filepath,
        "w",
        driver=driver,
        height=shape[0],
        width=shape[1],
        count=count,
        dtype=dtype,
    ):
        pass


def dataset_factories() -> Iterator[Callable]:
    yield tophu.BinaryFile
    yield lambda filepath, shape, dtype: tophu.HDF5Dataset(
        filepath, "/data", shape, dtype
    )
    yield lambda filepath, shape, dtype: tophu.RasterBand(
        filepath,
        width=shape[1],
        height=shape[0],
        dtype=dtype,
        driver="GTiff",
    )


class TestDatasets:
    @pytest.fixture(scope="class", params=dataset_factories())
    def dataset(self, request):
        shape = (1024, 512)
        dtype = np.int32
        with tempfile.NamedTemporaryFile() as f:
            dataset_factory = request.param
            yield dataset_factory(f.name, shape, dtype)

    def test_is_dataset_reader(self, dataset):
        # Check that each dataset satisfies the requirements of `DatasetReader`.
        assert isinstance(dataset, tophu.DatasetReader)

    def test_is_dataset_writer(self, dataset):
        # Check that each dataset satisfies the requirements of `DatasetWriter`.
        assert isinstance(dataset, tophu.DatasetWriter)

    def test_shape(self, dataset):
        assert dataset.shape == (1024, 512)

    def test_dtype(self, dataset):
        assert isinstance(dataset.dtype, np.dtype)
        assert dataset.dtype == np.int32

    def test_ndim(self, dataset):
        assert dataset.ndim == 2

    def test_setitem_getitem_roundtrip(self, dataset):
        # Test writing to & reading from each type of dataset.
        data = np.arange(20).reshape(4, 5)
        idx = np.s_[100:104, 200:205]
        dataset[idx] = data
        out = dataset[idx]
        assert np.all(out == data)

    def test_arraylike(self, dataset):
        # Check that each dataset can be converted to a `numpy.ndarray` object.
        arr = np.asarray(dataset)
        assert arr.shape == dataset.shape
        assert arr.dtype == dataset.dtype


class TestBinaryFile:
    def test_attrs(self):
        with tempfile.NamedTemporaryFile() as f:
            filepath = f.name
            shape = (100, 200)
            ndim = 2
            dtype = "f4"
            binary_file = tophu.BinaryFile(filepath, shape, dtype)
            assert binary_file.filepath == Path(filepath)
            assert binary_file.shape == shape
            assert binary_file.ndim == ndim
            assert binary_file.dtype == np.dtype(dtype)

    def test_create_file(self):
        # Check that `BinaryFile` creates the file if it didn't already exist.
        with tempfile.TemporaryDirectory() as d:
            filepath = Path(d) / "tmp.f8"
            assert not filepath.is_file()

            tophu.BinaryFile(filepath, shape=(3, 4, 5), dtype=np.float64)
            assert filepath.is_file()

    def test_read_file(self):
        with tempfile.NamedTemporaryFile() as f:
            # Write a byte string to a file.
            bytes = b"Hello world!"
            Path(f.name).write_bytes(bytes)

            # Open the file as a `BinaryFile` object.
            shape = 12
            dtype = np.uint8
            binary_file = tophu.BinaryFile(f.name, shape, dtype)

            # Read the contents and check that they match.
            assert np.asarray(binary_file).tobytes() == bytes

    def test_extend_file(self):
        with tempfile.NamedTemporaryFile() as f:
            # Create a small file.
            filepath = Path(f.name)
            filepath.write_text("Hello world!")
            assert filesize(filepath) == 12

            # Create a larger `BinaryFile` at the same location.
            shape = (100,)
            dtype = np.complex64
            tophu.BinaryFile(filepath, shape, dtype)

            # Check that the file size was extended to the expected size.
            assert filesize(filepath) == 800


class TestHDF5Dataset:
    def test_attrs(self):
        with tempfile.NamedTemporaryFile() as f:
            filepath = f.name
            datapath = "/path/to/dataset"
            shape = (100, 200)
            ndim = 2
            dtype = "f4"
            hdf5_dataset = tophu.HDF5Dataset(filepath, datapath, shape, dtype)
            assert hdf5_dataset.filepath == Path(filepath)
            assert hdf5_dataset.datapath == datapath
            assert hdf5_dataset.shape == shape
            assert hdf5_dataset.ndim == ndim
            assert hdf5_dataset.dtype == np.dtype(dtype)
            assert hdf5_dataset.chunks is None

    def test_create_file(self):
        # Check that `HDF5Dataset` creates the HDF5 file and dataset if they didn't
        # already exist.
        with tempfile.TemporaryDirectory() as d:
            filepath = Path(d) / "tmp.h5"
            assert not filepath.is_file()

            datapath = "/data"
            shape = (3, 4, 5)
            dtype = np.float64
            tophu.HDF5Dataset(filepath, datapath, shape, dtype)
            assert filepath.is_file()

            with h5py.File(filepath, "r") as f:
                assert datapath in f
                dataset = f[datapath]
                assert dataset.shape == shape
                assert dataset.dtype == dtype

    def test_create_dataset(self):
        # Check that `HDF5Dataset` creates the dataset if the HDF5 file already existed
        # but didn't contain the specified dataset.
        with tempfile.NamedTemporaryFile() as f:
            filepath = f.name
            with h5py.File(filepath, "w") as f:
                pass

            datapath = "/data"
            shape = (3, 4, 5)
            dtype = np.float64
            tophu.HDF5Dataset(filepath, datapath, shape, dtype)

            with h5py.File(filepath, "r") as f:
                assert datapath in f
                dataset = f[datapath]
                assert dataset.shape == shape
                assert dataset.dtype == dtype

    def test_existing_dataset(self):
        # Test creating an `HDF5Dataset` from an existing dataset in an HDF5 file.
        with tempfile.NamedTemporaryFile() as f:
            data = np.arange(20).reshape(4, 5)
            datapath = "/data"

            filepath = f.name
            with h5py.File(filepath, "w") as f:
                f[datapath] = data

            hdf5_dataset = tophu.HDF5Dataset(filepath, datapath)
            assert hdf5_dataset.shape == data.shape
            assert hdf5_dataset.dtype == data.dtype

            arr = np.asarray(hdf5_dataset)
            assert np.all(arr == data)

    def test_chunks(self):
        # Test creating a Dataset with chunked storage.
        with tempfile.NamedTemporaryFile() as f:
            chunks = (128, 129)
            hdf5_dataset = tophu.HDF5Dataset(
                filepath=f.name,
                datapath="/data",
                shape=(1024, 1024),
                dtype=np.int64,
                chunks=chunks,
            )
            assert hdf5_dataset.chunks == chunks

    def test_existing_dataset_chunks(self):
        with tempfile.NamedTemporaryFile() as f:
            filepath = f.name
            datapath = "/data"
            chunks = (128, 129)
            with h5py.File(filepath, "w") as f:
                f.create_dataset(
                    datapath,
                    shape=(1024, 1024),
                    dtype=np.int64,
                    chunks=chunks,
                )

            hdf5_dataset = tophu.HDF5Dataset(filepath, datapath)
            assert hdf5_dataset.chunks == chunks

    def test_bad_init_overload(self):
        errmsg = (
            r"^the supplied arguments don't match any valid overload of HDF5Dataset"
        )
        with pytest.raises(TypeError, match=errmsg):
            # Required parameter `dtype` is missing.
            tophu.HDF5Dataset(filepath="asdf.h5", datapath="/data", shape=(128, 128))


class TestRasterBand:
    def test_open_single_band(self):
        with tempfile.NamedTemporaryFile() as f:
            filepath = f.name
            shape = (1024, 512)
            dtype = np.float32
            driver = "GTiff"
            create_raster_dataset(
                filepath=filepath,
                driver=driver,
                shape=shape,
                count=1,
                dtype=dtype,
            )

            raster_band = tophu.RasterBand(filepath)
            assert raster_band.filepath == Path(filepath)
            assert raster_band.band == 1
            assert raster_band.driver == driver
            assert raster_band.shape == shape
            assert raster_band.dtype == dtype
            assert raster_band.ndim == 2

    def test_open_multi_band(self):
        with tempfile.NamedTemporaryFile() as f:
            filepath = f.name
            driver = "GTiff"
            shape = (1024, 512)
            dtype = np.int64
            create_raster_dataset(
                filepath=filepath,
                driver=driver,
                shape=shape,
                count=3,
                dtype=dtype,
            )

            raster_band = tophu.RasterBand(filepath, band=2)
            assert raster_band.filepath == Path(filepath)
            assert raster_band.band == 2
            assert raster_band.driver == driver
            assert raster_band.shape == shape
            assert raster_band.dtype == dtype
            assert raster_band.ndim == 2

    def test_missing_band(self):
        with tempfile.NamedTemporaryFile() as f:
            filepath = f.name
            create_raster_dataset(
                filepath=filepath,
                driver="GTiff",
                shape=(100, 100),
                count=3,
                dtype=np.float64,
            )

            errmsg = (
                r"^dataset contains \d+ raster bands: band index must be specified$"
            )
            with pytest.raises(ValueError, match=errmsg):
                tophu.RasterBand(filepath)

    @pytest.mark.parametrize("band", [0, 4])
    def test_bad_band(self, band):
        with tempfile.NamedTemporaryFile() as f:
            filepath = f.name
            create_raster_dataset(
                filepath=filepath,
                driver="GTiff",
                shape=(100, 100),
                count=3,
                dtype=np.float64,
            )

            errmsg = r"^band index \d+ out of range: dataset contains \d+ raster bands$"
            with pytest.raises(IndexError, match=errmsg):
                tophu.RasterBand(filepath, band=band)

    @pytest.mark.parametrize("dtype", valid_gdal_dtypes())
    def test_create_dataset(self, dtype):
        with tempfile.NamedTemporaryFile() as f:
            filepath = Path(f.name)
            height = 1024
            width = 512
            driver = "GTiff"

            raster_band = tophu.RasterBand(
                filepath, width, height, dtype, driver=driver
            )
            assert raster_band.filepath == filepath
            assert raster_band.band == 1
            assert raster_band.shape == (height, width)
            assert raster_band.dtype == dtype
            assert raster_band.driver == driver
            assert raster_band.crs is None

    @pytest.mark.parametrize("ext", [".tif", ".tiff"])
    def test_infer_driver_from_extension(self, ext):
        with tempfile.TemporaryDirectory() as d:
            filepath = Path(d) / ("tmp" + ext)
            raster_band = tophu.RasterBand(
                filepath,
                width=100,
                height=100,
                dtype=np.float32,
            )
            assert raster_band.driver == "GTiff"

    def test_crs(self):
        with tempfile.NamedTemporaryFile() as f:
            raster_band = tophu.RasterBand(
                f.name,
                width=100,
                height=100,
                dtype=np.float32,
                driver="GTiff",
                crs="epsg:4326",
            )

            crs = raster_band.crs
            assert isinstance(crs, rasterio.crs.CRS)
            assert crs.to_epsg() == 4326

    def test_transform(self):
        with tempfile.NamedTemporaryFile() as f:
            transform = (
                rasterio.transform.Affine.identity()
                .scale(10.0)
                .translation(100.0, 200.0)
            )
            raster_band = tophu.RasterBand(
                f.name,
                width=100,
                height=100,
                dtype=np.float32,
                driver="GTiff",
                transform=transform,
            )
            assert raster_band.transform == transform

    def test_bad_init_overload(self):
        errmsg = r"^the supplied arguments don't match any valid overload of RasterBand"
        with pytest.raises(TypeError, match=errmsg):
            # Required parameter `dtype` is missing.
            tophu.RasterBand(filepath="asdf.tif", width=128, height=128)
