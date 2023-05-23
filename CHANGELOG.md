# Unreleased

**Added**

- Support for upsampling N-D arrays (FFT-based & nearest neighbor)
- Basic support for multilooking
- Band pass FIR filter implementation using the optimal equiripple method
- Tile manager class
- Abstract interface to "plug-in" unwrapping algorithms
- Unwrapping via SNAPHU, PHASS, and ICU
- Baseline multi-scale unwrapping implementation

**Changed**

**Deprecated**

**Removed**

**Fixed**

**Dependencies**

- Require dask>=2022.05.1
- Require isce3>=0.12
- Require numpy>=1.21
- Require python>=3.8
- Require rasterio>=1.3
- Require scipy>=1.5
