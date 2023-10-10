Usage
#####

Basic usage with raster files
=============================

.. code-block:: python

    from tophu import RasterBand, SnaphuUnwrap, multiscale_unwrap
    import rasterio as rio

    # Create `RasterBand`s from the input interferogram and coherence.
    igram = RasterBand(ifg_filename)
    coherence = RasterBand(corr_filename)

    # Create the output raster bands.
    # Start by copying the input geographic metadata.
    with rio.open(ifg_filename) as src:
        profile = src.profile

    # The unwrapped phase will be float32.
    profile["dtype"] = np.float32
    profile["driver"] = "GTiff"
    unw = RasterBand("unwrapped_phase.unw.tif", **profile)

    # Create the connected component labels raster.
    profile["dtype"] = np.uint16
    conncomp = RasterBand("connected_components.tif", **profile)

    # Choose which unwrapper we will use.
    # Here we pick SNAPHU.
    unwrap_callback = SnaphuUnwrap(
        cost="smooth",
        init_method="mst",
    )

    # Set the number of looks used to form the coherence.
    nlooks = 40

    # Choose the tiling scheme and the downsample factor for the coarse unwrap.
    ntiles = (2, 2)
    downsample_factor = (3, 3)

    # Run the multiscale unwrapping function.
    multiscale_unwrap(
        unw,
        conncomp,
        igram,
        coherence,
        nlooks=nlooks,
        unwrap_func=unwrap_callback,
        downsample_factor=downsample_factor,
        ntiles=ntiles,
    )
