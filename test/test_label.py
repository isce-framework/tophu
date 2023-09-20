from __future__ import annotations

import itertools

import dask.array as da
import numpy as np
import numpy.testing as npt
import pytest
import scipy.ndimage
from numpy.typing import NDArray

import tophu
from tophu._label import NO_OVERLAPPING_LABEL, find_max_overlapping_labels, relabel


def random_binary_mask(
    shape: tuple[int, ...],
    *,
    density: float,
    seed: int | None = None,
) -> NDArray[np.bool_]:
    """
    Create a random binary mask.

    Parameters
    ----------
    shape : tuple of int
        The shape of the output array.
    density : float
        The approximate density of nonzero values in the mask. Must be between 0 and 1.
    seed : int or None, optional
        Seed for initializing pseudo-random number generator state. Must be nonnegative.
        If None, then the generator will be initialized randomly. Defaults to None.

    Returns
    -------
    mask : numpy.ndarray
        A randomly generated binary mask.
    """
    if (density < 0.0) or (density > 1.0):
        raise ValueError("density must be >= 0 and <= 1")

    # Setup a pseudo-random number generator.
    rng = np.random.default_rng(seed)

    # Generate a random binary mask by uniformly sampling the interval [0, 1] and
    # thresholding.
    return rng.uniform(size=shape) < density


def random_conncomp_labels(
    shape: tuple[int, ...],
    *,
    density: float,
    seed: int | None = None,
) -> NDArray[np.integer]:
    """
    Create a random array of connected component labels.

    Each connected component is assigned a unique positive integer label. Pixels not
    belonging to any connected component are labeled 0.

    Parameters
    ----------
    shape : tuple of int
        The shape of the output array.
    density : float
        The approximate density of nonzero values in the output array. Must be between 0
        and 1.
    seed : int or None, optional
        Seed for initializing pseudo-random number generator state. Must be nonnegative.
        If None, then the generator will be initialized randomly. Defaults to None.

    Returns
    -------
    conncomp : numpy.ndarray
        The output array of randomly generated connected component labels.
    """
    mask = random_binary_mask(shape, density=density, seed=seed)
    conncomp, _ = scipy.ndimage.label(mask)
    return conncomp


class TestFindMaxOverlappingLabels:
    def test_fully_overlapping(self):
        # A connected components array with 3 rectangular-shaped components.
        cc1 = np.zeros((128, 128), np.uint32)
        cc1[10:20, 10:20] = 1
        cc1[50:55, 5:25] = 2
        cc1[80:100, 80:120] = 3

        # Second connected components array is the same as the first
        cc2 = cc1.copy()

        label_mapping = find_max_overlapping_labels(cc1, cc2, min_overlap=1.0)

        # The output dict should map each original CC label to itself.
        assert label_mapping == {1: 1, 2: 2, 3: 3}

    def test_multiple_overlapping_labels(self):
        shape = (128, 128)
        dtype = np.uint32

        # Initial CC array just contains one big component.
        cc1 = np.ones(shape, dtype)

        # The second CC array contains 3 rectangular-shaped components with different
        # sizes.
        cc2 = np.zeros(shape, dtype)
        cc2[10:20, 10:20] = 1
        cc2[50:55, 5:25] = 2
        cc2[80:100, 80:120] = 3

        label_mapping = find_max_overlapping_labels(cc1, cc2, min_overlap=1e-6)

        # Check that the mapped-to component label is the one with the most overlapping
        # area (i.e. the largest component in the second array).
        assert label_mapping == {1: 3}

    def test_no_overlap(self):
        shape = (128, 128)
        dtype = np.uint32

        # A connected components array with 3 rectangular-shaped components.
        cc1 = np.zeros(shape, dtype)
        cc1[10:20, 10:20] = 1
        cc1[50:55, 5:25] = 2
        cc1[80:100, 80:120] = 3

        # Another connected components array with no components that overlap with the
        # first array's components.
        cc2 = np.zeros(shape, dtype)
        cc2[10:20, 21:31] = 4
        cc2[50:55, 26:50] = 5
        cc2[60:79, 80:120] = 6

        label_mapping = find_max_overlapping_labels(cc1, cc2, min_overlap=1e-6)

        # Check that each initial label is mapped to `NO_OVERLAPPING_LABEL`.
        expected = {
            1: NO_OVERLAPPING_LABEL,
            2: NO_OVERLAPPING_LABEL,
            3: NO_OVERLAPPING_LABEL,
        }
        assert label_mapping == expected

    def test_insufficient_overlap(self):
        shape = (128, 128)
        dtype = np.uint32

        # Initial CC array just contains one big component.
        cc1 = np.ones(shape, dtype)

        # Second CC array contains one component whose intersection with the first
        # component is 49% of its total area.
        cc2 = np.zeros(shape, dtype)
        cc2[:5] = 1
        cc2[0, 0] = 0

        # Require CCs to overlap by >= 50% to be considered overlapping.
        label_mapping = find_max_overlapping_labels(cc1, cc2, min_overlap=0.5)

        # Check that the initial label is mapped to `NO_OVERLAPPING_LABEL`.
        assert label_mapping == {1: NO_OVERLAPPING_LABEL}

    def test_overlap_ratio(self):
        shape = (128, 128)
        dtype = np.uint32

        # Initial CC array contains one small rectangular-shaped component.
        cc1 = np.zeros(shape, dtype)
        cc1[50:60, 20:30] = 1

        # Second CC array contains one much larger rectangular-shaped component.
        # The intersection between the two components is 50% of the initial component's
        # area, but a much smaller percentage of the second component's area.
        cc2 = np.zeros(shape, dtype)
        cc2[10:120, 25:100] = 2

        # Require CCs to overlap by >= 50% to be considered overlapping.
        label_mapping = find_max_overlapping_labels(cc1, cc2, min_overlap=0.5)

        # Check that the components are sufficiently overlapping.
        assert label_mapping == {1: 2}

    def test_all_zeros(self):
        shape = (128, 128)
        dtype = np.uint32

        # Initial CC array is all zeros.
        cc1 = np.zeros(shape, dtype)

        # The second CC array contains 3 rectangular-shaped components with different
        # sizes.
        cc2 = np.zeros(shape, dtype)
        cc2[10:20, 10:20] = 1
        cc2[50:55, 5:25] = 2
        cc2[80:100, 80:120] = 3

        label_mapping = find_max_overlapping_labels(cc1, cc2)

        # Check that the label mapping is an empty dict.
        assert label_mapping == {}

    def test_shape_mismatch(self):
        cc1 = random_conncomp_labels(shape=(100, 100), density=0.6)
        cc2 = random_conncomp_labels(shape=(100, 101), density=0.6)

        errmsg = (
            r"^shape mismatch: input connected components arrays must have the same"
            r" shape$"
        )
        with pytest.raises(ValueError, match=errmsg):
            find_max_overlapping_labels(cc1, cc2)

    def test_bad_min_overlap(self):
        cc1 = random_conncomp_labels(shape=(128, 128), density=0.6)
        cc2 = random_conncomp_labels(shape=(128, 128), density=0.6)

        with pytest.raises(ValueError, match=r"^min overlap must be > 0"):
            find_max_overlapping_labels(cc1, cc2, min_overlap=0.0)

        with pytest.raises(ValueError, match=r"^min overlap must be <= 1"):
            find_max_overlapping_labels(cc1, cc2, min_overlap=1.0 + 1e-6)


def test_relabel():
    # A connected components array with 3 rectangular-shaped components.
    conncomp = np.zeros((128, 128), np.uint32)
    conncomp[10:20, 10:20] = 1
    conncomp[50:55, 5:25] = 2
    conncomp[80:100, 80:120] = 3

    # Make a copy of the input connected components array.
    conncomp_orig = conncomp.copy()

    # Define a mapping from input labels to output labels.
    label_mapping = {1: 4, 2: 1, 3: 3}

    # Relabel.
    relabeled = relabel(conncomp, label_mapping)

    # Check the output labels for each CC.
    for label in [1, 2, 3]:
        mask1 = conncomp == label
        mask2 = relabeled == label_mapping[label]
        npt.assert_array_equal(mask1, mask2)

    # Check that the input array was not modified.
    npt.assert_array_equal(conncomp, conncomp_orig)


def iterchunks(arr: da.Array) -> np.ndarray:
    """Iterate over blocks of a Dask array."""
    for ix in itertools.product(*map(range, arr.blocks.shape)):
        yield np.asanyarray(arr.blocks[ix])


class TestRelabelHiresConncomps:
    def test_deduplicate_labels(self):
        shape = (200, 200)
        chunksize = (100, 100)
        dtype = np.uint32

        # Each tile contains a single connected component, all with the same label.
        tiled_conncomp = da.ones(shape=shape, dtype=dtype, chunks=chunksize)

        # The coarse CC array contains four different components, one filling each
        # quadrant of the array.
        coarse_conncomp_ = np.zeros(shape=shape, dtype=dtype)
        coarse_conncomp_[:100, :100] = 1
        coarse_conncomp_[:100, 100:] = 2
        coarse_conncomp_[100:, :100] = 3
        coarse_conncomp_[100:, 100:] = 4
        coarse_conncomp = da.from_array(coarse_conncomp_, chunks=chunksize)

        # Relabel.
        relabeled = tophu.relabel_hires_conncomps(tiled_conncomp, coarse_conncomp)

        # After relabeling, there should be no common labels between different tiles.
        # Get the set of unique labels within each tile and check that each pair of sets
        # is disjoint (has no common elements).
        unique_block_labels = [set(np.unique(block)) for block in iterchunks(relabeled)]
        for set1, set2 in itertools.combinations(unique_block_labels, r=2):
            assert set1.isdisjoint(set2)

    def test_no_coarse_conncomps(self):
        # Each tile contains a single connected component, all with the same label.
        tiled_conncomp = da.ones(shape=(200, 200), dtype=np.uint32, chunks=(100, 100))

        # The coarse connected components array contains no components.
        coarse_conncomp = da.zeros_like(tiled_conncomp)

        # Relabel.
        relabeled = tophu.relabel_hires_conncomps(tiled_conncomp, coarse_conncomp)

        # After relabeling, there should be no common labels between different tiles.
        # Get the set of unique labels within each tile and check that each pair of sets
        # is disjoint (has no common elements).
        unique_block_labels = [set(np.unique(block)) for block in iterchunks(relabeled)]
        for set1, set2 in itertools.combinations(unique_block_labels, r=2):
            assert set1.isdisjoint(set2)

    def test_merge_labels(self):
        # Create random CCs across all tiles, each with a unique label.
        shape = (128, 128)
        tiled_conncomp_ = random_conncomp_labels(shape, density=0.6, seed=1234)
        tiled_conncomp = da.from_array(tiled_conncomp_, chunks=(65, 65))

        # The coarse CC array is filled by a single component.
        coarse_conncomp = da.ones_like(tiled_conncomp)

        # Relabel.
        relabeled = tophu.relabel_hires_conncomps(tiled_conncomp, coarse_conncomp)

        # Each CC should have the same label after relabeling.
        mask = tiled_conncomp != 0
        assert da.all(relabeled[mask] == 1)
        assert da.all(relabeled[~mask] == 0)

    def test_output_natural_numbers(self):
        # Generate an initial set of CCs but add 100 to each label.
        shape = (128, 128)
        tiled_conncomp_ = random_conncomp_labels(shape, density=0.6, seed=1234)
        tiled_conncomp_[tiled_conncomp_ != 0] += 100
        tiled_conncomp = da.from_array(tiled_conncomp_, chunks=(64, 64))

        # Coarse CCs are identical to tiled CCs.
        coarse_conncomp = tiled_conncomp.copy()

        # Relabel.
        relabeled = tophu.relabel_hires_conncomps(tiled_conncomp, coarse_conncomp)
        relabeled = relabeled.compute()

        # The new connected components should be labeled [1, 2, ..., N], where N is the
        # total number of components, regardless of the initial labels.
        nlabels = len(tophu.unique_nonzero_integers(relabeled))
        unique_labels = np.unique(relabeled)
        npt.assert_array_equal(unique_labels, np.arange(nlabels + 1))

    def test_shape_mismatch(self):
        tiled_conncomp_ = random_conncomp_labels(shape=(100, 100), density=0.6)
        coarse_conncomp_ = random_conncomp_labels(shape=(100, 101), density=0.6)

        tiled_conncomp = da.from_array(tiled_conncomp_, chunks=(128, 128))
        coarse_conncomp = da.from_array(coarse_conncomp_, chunks=(128, 128))

        errmsg = (
            r"^shape mismatch: the high-res and low-res connected components arrays"
            r" must have the same shape$"
        )
        with pytest.raises(ValueError, match=errmsg):
            tophu.relabel_hires_conncomps(tiled_conncomp, coarse_conncomp)

    def test_chunksize_mismatch(self):
        tiled_conncomp_ = random_conncomp_labels(shape=(100, 100), density=0.6)
        coarse_conncomp_ = random_conncomp_labels(shape=(100, 100), density=0.6)

        tiled_conncomp = da.from_array(tiled_conncomp_, chunks=(50, 50))
        coarse_conncomp = da.from_array(coarse_conncomp_, chunks=(50, 51))

        errmsg = (
            r"^the high-res and low-res connected components arrays must have the same"
            r" chunk sizes$"
        )
        with pytest.raises(ValueError, match=errmsg):
            tophu.relabel_hires_conncomps(tiled_conncomp, coarse_conncomp)
