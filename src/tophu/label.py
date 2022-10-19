import itertools
from typing import Set, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage

from .tile import TiledPartition
from .union_find import DisjointSetForest

__all__ = [
    "deduplicate_labels",
    "merge_equivalent_labels",
    "unique_nonzero_integers",
]


NDSlice = Tuple[slice, ...]


def deduplicate_labels(
    conncomp: NDArray[np.unsignedinteger],
    tiles: TiledPartition,
) -> None:
    """
    De-duplicate connected component labels from different tiles.

    Connected component labels within each tile are updated such that no tile has any
    label in common with any other tile.

    Only nonzero labels are modified. Zero-valued pixels are treated as masked out (i.e.
    not part of any connected component).

    The `conncomp` array is modified in-place.

    Parameters
    ----------
    conncomp : numpy.ndarray
        The input connected component labels.
    tiles : TiledPartition
        A set of tiles that spans the `conncomp` array. Each tile is represented by an
        index expression (i.e. a tuple of `slice` objects) that can be used to access
        a corresponding block of data from the partitioned array.

    See Also
    --------
    merge_equivalent_labels
    """
    # Keep track of the largest connected component label encountered while looping over
    # tiles.
    max_label = 0
    for tile in tiles:
        # Get a mask of elements within the tile with nonzero connected component
        # labels.
        mask = conncomp[tile] != 0

        # Update connected component labels by adding the largest label value from among
        # all previously encountered tiles. Zero-valued labels are not changed. The
        # input connected component labels array is modified in-place.
        conncomp[tile][mask] += max_label

        # Get the new maximum connected component label.
        max_label = max(max_label, np.max(conncomp[tile]))


def expand_tile(tile: NDSlice, margin: int) -> NDSlice:
    """
    Expand a tile by adding some margin to each border.

    Parameters
    ----------
    tile : tuple of slice
        A tuple of slice objects representing a multi-dimensional indexing expression,
        such as that returned by `numpy.index_exp`.
    margin : int
        The number of samples to add to each border of the tile.

    Returns
    -------
    expanded_tile : tuple of slice
        The expanded indexing expression.
    """
    if margin < 0:
        raise ValueError("margin must be >= 0")

    def expand_slice(s: slice) -> slice:
        if s.start is None:
            new_start = None
        else:
            # Subtract `margin` from the start index of the slice. We need to be careful
            # to avoid changing the sign of the index since negative indices in Python
            # typically wrap around.
            sign_change = (s.start >= 0) and (s.start < margin)
            new_start = 0 if sign_change else s.start - margin

        if s.stop is None:
            new_stop = None
        else:
            # Add `margin` to the stop index of the slice. Avoid changing the sign of
            # the index.
            sign_change = (s.stop < 0) and (s.stop >= -margin)
            new_stop = None if sign_change else s.stop + margin

        return slice(new_start, new_stop, s.step)

    return tuple([expand_slice(s) for s in tile])


def unique_nonzero_integers(x: ArrayLike, /) -> Set:
    """
    Find unique nonzero elements in the input array.

    Parameters
    ----------
    x : array_like
        An array of integers.

    Returns
    -------
    s : set
        The set of unique nonzero values.
    """
    return set(np.unique(x)) - {0}


def find_connected_labels(
    label: int,
    conncomp: NDArray[np.unsignedinteger],
    *,
    connectivity: int = 1,
) -> Set[int]:
    """
    Find equivalent labels based on pixel connectivity.

    Determine the set of equivalent labels to the specified label. Two nonzero labels
    are considered equivalent if there is any connected path between pixels with these
    labels.

    Zero-valued pixels are ignored.

    Parameters
    ----------
    label : int
        The query label.
    conncomp : numpy.ndarray
        A two-dimensional array of positive integer labels associated with connected
        components. Some connected components may have more than one label. Zero-valued
        pixels are treated as masked-out (i.e. not part of any connected component).
    connectivity : int, optional
        Determines which pixels are considered connected. Valid values are 1 or 2. If
        `connectivity=1`, only a pixel's 4 orthogonal neighbors are considered
        connected. If `connectivity=2`, a pixel's 8 orthogonal and diagonal neighbors
        are considered connected. (default: 1)

    Returns
    -------
    equivalent_labels : set of int
        A set of connected component labels that are equivalent to the specified label.
    """
    if connectivity not in {1, 2}:
        raise ValueError("connectivity must be 1 or 2")

    # Get a mask of pixels in `conncomp` with value equal to `label`.
    mask = conncomp == label

    # Dilate the masked region to include bordering pixels.
    struct = ndimage.generate_binary_structure(conncomp.ndim, connectivity)
    dilated_mask = ndimage.binary_dilation(mask, struct)

    # Get a mask of only pixels bordering the labeled region.
    edge_mask = dilated_mask ^ mask

    # Get the set of unique nonzero values in the bordering pixels.
    return unique_nonzero_integers(conncomp[edge_mask])


def merge_equivalent_labels(
    conncomp: NDArray[np.unsignedinteger],
    tiles: TiledPartition,
    *,
    connectivity: int = 1,
) -> NDArray[np.unsignedinteger]:
    r"""
    Merge equivalent connected component labels.

    Find connected components in the input array with multiple labels and relabel them
    with a single unique label. After relabeling, each individual connected region is
    assigned a new positive integer label in [1, 2, ..., N], where N is the total number
    of disjoint connected components.

    Zero-valued pixels are treated as masked out (i.e. not part of any connected
    component). They are not relabeled.

    A two-pass algorithm is used to determine the new connected components\
    :footcite:p:`fiorio:1996`. In the first phase, the initial labels are scanned and
    label equivalence information is stored in
    a `Union-Find data structure
    <https://en.wikipedia.org/wiki/Disjoint-set_data_structure>`_. Each set of
    equivalent labels is associated with a unique final label. A second pass through the
    data assigns each pixel its final label.

    Parameters
    ----------
    conncomp : numpy.ndarray
        The initial connected component labels. A two-dimensional array of nonnegative
        integer values. Some connected components may have more than one label.
        Zero-valued pixels are treated as masked-out (i.e. not part of any connected
        component).
    tiles : TiledPartition
        A set of tiles that spans the `conncomp` array. Each tile is represented by an
        index expression (i.e. a tuple of `slice` objects) that can be used to access
        a corresponding block of data from the partitioned array.
    connectivity : int, optional
        Determines which pixels are considered connected. Valid values are 1 or 2. If
        `connectivity=1`, only a pixel's 4 orthogonal neighbors are considered
        connected. If `connectivity=2`, a pixel's 8 orthogonal and diagonal neighbors
        are considered connected. (default: 1)

    Returns
    -------
    new_conncomp : numpy.ndarray
        The output array of updated connected component labels.

    See Also
    --------
    deduplicate_labels
    """
    # Initialize the union-find data structure.
    forest: DisjointSetForest[int] = DisjointSetForest()

    # First pass: compute label equivalences.
    for tile in tiles:
        # Expand the tile by one pixel along each border so that it overlaps with
        # neighboring tiles.
        expanded_tile = expand_tile(tile, 1)

        # Find unique nonzero labels and add them as new disjoint sets to the union-find
        # data structure.
        unique_nonzero_labels = unique_nonzero_integers(conncomp[expanded_tile])
        forest.add_items(unique_nonzero_labels)

        # Loop over the set of labels.
        for label in unique_nonzero_labels:
            # Find any equivalent labels (i.e. labels associated with any neighboring
            # connected pixel).
            equivalent_labels = find_connected_labels(
                label,
                conncomp[expanded_tile],
                connectivity=connectivity,
            )

            # Merge equivalent labels.
            for equivalent_label in equivalent_labels:
                forest.union(label, equivalent_label)

    # Flatten the forest so that each node in each tree is a direct child of its root
    # node in order to improve efficiency of tree traversal.
    forest.flatten()

    # After the first pass, each disjoint set of equivalent labels is now associated
    # with a new unique label (represented by the root of each tree within the
    # disjoint-set forest). However, these new labels are not necessarily consecutive
    # integers, nor are they necessarily bounded by the total number of connected
    # components.

    # Define a mapping from the set of representative labels (root nodes) in the
    # disjoint-set forest to the sequence of natural numbers 1 through N (where N is the
    # total number of disjoint sets), which will form the final set of labels.
    final_labels = dict(zip(forest.roots(), itertools.count(1)))

    # Init output connected component labels array.
    new_conncomp = np.zeros_like(conncomp)

    # Second pass: relabel connected components.
    for tile in tiles:
        # Get the set of unique nonzero labels within the tile.
        unique_nonzero_labels = unique_nonzero_integers(conncomp[tile])

        for label in unique_nonzero_labels:
            # Get a mask of pixels in the tile with value equal to `label`.
            mask = conncomp[tile] == label

            # Relabel pixels with their new assigned label.
            root = forest.find(label)
            new_conncomp[tile][mask] = final_labels[root]

    return new_conncomp
