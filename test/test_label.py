import itertools
from typing import Optional, Tuple

import numpy as np
import pytest
import scipy.ndimage
from numpy.typing import NDArray

import tophu


def random_binary_mask(
    shape: Tuple[int, ...],
    *,
    density: float,
    seed: Optional[int] = None,
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
        If None, then the generator will be initialized randomly. (default: None)

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
    shape: Tuple[int, ...],
    *,
    density: float,
    seed: Optional[int] = None,
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
        If None, then the generator will be initialized randomly. (default: None)

    Returns
    -------
    conncomp : numpy.ndarray
        The output array of randomly generated connected component labels.
    """
    mask = random_binary_mask(shape, density=density, seed=seed)
    conncomp, _ = scipy.ndimage.label(mask)
    return conncomp


class TestDeduplicateLabels:
    def test1(self):
        # Initially, each pixel belongs to a single connected component with label 1.
        shape = (1024, 1024)
        conncomp = np.ones(shape, dtype=np.uint32)

        # De-duplicate labels from different tiles.
        tiles = tophu.TiledPartition(shape, ntiles=(3, 3))
        tophu.deduplicate_labels(conncomp, tiles)

        # After re-labeling, there should be no common labels between different tiles.
        # Get the set of unique labels within each tile and check that each pair of sets
        # is disjoint (has no common elements).
        tile_label_sets = [set(np.unique(conncomp[tile])) for tile in tiles]
        for set1, set2 in itertools.combinations(tile_label_sets, r=2):
            assert set1.isdisjoint(set2)

    def test2(self):
        length, width = 512, 512

        # Create two connected components: one outer rectangular annulus and one inner
        # rectangle.
        conncomp1_mask = np.ones((length, width), dtype=np.bool_)
        conncomp1_mask[64:-64, 64:-64] = False

        conncomp2_mask = np.zeros((length, width), dtype=np.bool_)
        conncomp2_mask[192:-192, 192:-192] = True

        conncomp = np.zeros((length, width), dtype=np.uint32)
        conncomp[conncomp1_mask] = 1
        conncomp[conncomp2_mask] = 2

        # Partition the connected components array into a 2x2 grid of tiles such that
        # each connected component spans all 4 tiles.
        tiles = tophu.TiledPartition((length, width), ntiles=(2, 2))

        # De-duplicate labels from different tiles.
        tophu.deduplicate_labels(conncomp, tiles)

        # After re-labeling, there should be no common labels between different tiles
        # (except for masked-out pixels with label 0). Get the set of unique nonzero
        # labels within each tile and check that each pair of sets is disjoint (has no
        # common elements).
        tile_label_sets = [
            tophu.unique_nonzero_integers(conncomp[tile]) for tile in tiles
        ]
        for set1, set2 in itertools.combinations(tile_label_sets, r=2):
            assert set1.isdisjoint(set2)

        # Since both connected components spanned all 4 tiles, the relabeled connected
        # components should each consist of 4 unique labels.
        conncomp1_labels = set(np.unique(conncomp[conncomp1_mask]))
        conncomp2_labels = set(np.unique(conncomp[conncomp2_mask]))
        assert len(conncomp1_labels) == 4
        assert len(conncomp2_labels) == 4
        assert conncomp1_labels.isdisjoint(conncomp2_labels)

        # Each masked-out pixel should still be labeled 0.
        invalid_mask = ~(conncomp1_mask | conncomp2_mask)
        assert np.all(conncomp[invalid_mask] == 0)

    def test3(self):
        # Simulate a randomized array of connected component labels.
        shape = (128, 128)
        conncomp = random_conncomp_labels(shape, density=0.6)

        # De-duplicate labels from different tiles.
        tiles = tophu.TiledPartition(shape, ntiles=(2, 2))
        tophu.deduplicate_labels(conncomp, tiles)

        # After re-labeling, there should be no common labels between different tiles
        # (except for masked-out pixels with label 0). Get the set of unique nonzero
        # labels within each tile and check that each pair of sets is disjoint (has no
        # common elements).
        tile_label_sets = [
            tophu.unique_nonzero_integers(conncomp[tile]) for tile in tiles
        ]
        for set1, set2 in itertools.combinations(tile_label_sets, r=2):
            assert set1.isdisjoint(set2)


class TestMergeEquivalentLabels:
    def test(self):
        length, width = 1280, 256

        # Create two connected components: one tall "U"-shaped component and one
        # rectangular-shaped component.
        conncomp1_mask = np.zeros((length, width), dtype=np.bool_)
        conncomp1_mask[50:1000, 50:100] = True
        conncomp1_mask[50:1000, -100:-50] = True
        conncomp1_mask[1000:1050, 50:-50] = True

        conncomp2_mask = np.zeros((length, width), dtype=np.bool_)
        conncomp2_mask[1100:-50, 50:-50] = True

        conncomp = np.zeros((length, width), dtype=np.uint32)
        conncomp[conncomp1_mask] = 1
        conncomp[conncomp2_mask] = 3

        # Update the connected component labels in each tile so that each component is
        # associated with multiple redundant labels.
        tiles = tophu.TiledPartition((length, width), ntiles=(5, 2))
        for i, tile in enumerate(tiles):
            nonzero_mask = conncomp[tile] != 0
            conncomp[tile][nonzero_mask] += i

        # Merge equivalent labels.
        new_conncomp = tophu.merge_equivalent_labels(conncomp, tiles)

        # Checks whether every value in the input array is exactly the same. The array
        # must not be empty.
        def all_same(x: NDArray) -> bool:
            return np.all(x == x.flat[0])

        # After relabeling, all pixels within each connected component should have the
        # same label.
        for mask in [conncomp1_mask, conncomp2_mask]:
            assert all_same(new_conncomp[mask])

        # The two connected components should have different labels.
        conncomp1_labels = set(np.unique(new_conncomp[conncomp1_mask]))
        conncomp2_labels = set(np.unique(new_conncomp[conncomp2_mask]))
        assert conncomp1_labels.isdisjoint(conncomp2_labels)

        # Outside of the two connected components, all pixels should be filled with
        # zeros.
        invalid_mask = ~(conncomp1_mask | conncomp2_mask)
        assert np.all(new_conncomp[invalid_mask] == 0)

    def test_output_labels(self):
        # Generate an initial set of connected components but add 100 to each connected
        # component label.
        shape = (128, 128)
        conncomp = random_conncomp_labels(shape, density=0.6)
        conncomp[conncomp != 0] += 100

        # Relabel.
        tiles = tophu.TiledPartition(shape, ntiles=(3, 3))
        new_conncomp = tophu.merge_equivalent_labels(conncomp, tiles)

        # The new connected components should be labeled [1, 2, ..., N], where N is the
        # total number of connected components, regardless of the initial connected
        # component labels.
        nlabels = len(tophu.unique_nonzero_integers(conncomp))
        assert np.all(np.unique(new_conncomp) == np.arange(nlabels + 1))

    def test_connectivity(self):
        length, width = 128, 128
        conncomp = np.zeros((length, width), dtype=np.uint32)

        # Divide the connected component labels array into quadrants. Fill the
        # upper-left quadrant with ones and the lower-right quadrant with twos. The
        # remaining quadrants are filled with zeros.
        halflength = length // 2
        halfwidth = width // 2
        conncomp[:halflength, :halfwidth] = 1
        conncomp[halflength:, halfwidth:] = 2

        tiles = tophu.TiledPartition((length, width), ntiles=(2, 2))

        # With `connectivity=1`, the two nonzero quadrants are considered disconnected,
        # so `merge_equivalent_labels()` has no effect.
        new_conncomp = tophu.merge_equivalent_labels(conncomp, tiles, connectivity=1)
        assert np.all(new_conncomp == conncomp)

        # But with `connectivity=2`, the two nonzero quadrants are connected via a
        # single pair of diagonal neighbors, so they should be merged into a single
        # connected component.
        new_conncomp = tophu.merge_equivalent_labels(conncomp, tiles, connectivity=2)
        assert np.all(new_conncomp == np.minimum(conncomp, 1))

    def test_bad_connectivity(self):
        shape = (128, 128)
        conncomp = random_conncomp_labels(shape, density=0.6)
        tiles = tophu.TiledPartition(shape, ntiles=(2, 2))

        # Check that an error is raised if the value of `connectivity` is invalid.
        with pytest.raises(ValueError):
            tophu.merge_equivalent_labels(conncomp, tiles, connectivity=3)
