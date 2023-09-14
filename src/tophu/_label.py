from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any

import dask.array as da
import numpy as np
from numpy.typing import NDArray

from ._util import get_all_unique_values, mode, unique_nonzero_integers

__all__ = [
    "relabel_hires_conncomps",
]


# A constant used to identify high-res connected components that don't overlap with any
# low-res component.
NO_OVERLAPPING_LABEL = -1


def find_max_overlapping_labels(
    src_conncomp: NDArray[np.unsignedinteger],
    dst_conncomp: NDArray[np.unsignedinteger],
    *,
    min_overlap: float = 0.5,
) -> dict[int, int]:
    """
    Find overlapping connected components.

    Given two sets of connected component labels, find the labels in the second set that
    most overlap with each label in the first set. That is, for each unique label in
    `src_conncomp`, compute the label from `dst_conncomp` that it has the largest
    intersecting area with, if any such label exists.

    The ratio of the intersecting area to the area of the original component must be at
    least `min_overlap` for the two components to be considered overlapping. The special
    constant `NO_OVERLAPPING_LABEL` is used to identify labels from `src_conncomp` that
    did not sufficiently overlap with any connected component from `dst_conncomp`.

    Zero-valued elements are not considered to be members of any connected component.

    Parameters
    ----------
    src_conncomp : numpy.ndarray
        The initial set of connected component labels. An array of nonnegative integers.
    dst_conncomp : numpy.ndarray
        The second set of connected component labels. An array of nonnegative integers
        with the same shape as `src_conncomp`.
    min_overlap : float, optional
        Minimum intersection between components in order to be considered overlapping,
        as a fraction of the area of the component from `src_conncomp`. Must be in the
        range (0, 1]. Defaults to 0.5.

    Returns
    -------
    overlapping_labels : dict
        A mapping from each unique label in `src_conncomp` to the label in
        `dst_conncomp` that it most overlapped with, or to `NO_OVERLAPPING_LABEL` if no
        connected component was found that satisfied the minimum overlap threshold.
    """
    if dst_conncomp.shape != src_conncomp.shape:
        raise ValueError(
            "shape mismatch: input connected components arrays must have the same shape"
        )

    if min_overlap <= 0.0:
        raise ValueError(f"min overlap must be > 0, got {min_overlap}")
    if min_overlap > 1.0:
        raise ValueError(f"min overlap must be <= 1, got {min_overlap}")

    # Get the set of unique labels in the first array of connected components (CCs).
    src_labels = unique_nonzero_integers(src_conncomp)

    # Get a mask of nonzero values in the second array of CCs.
    dst_nonzero = dst_conncomp != 0

    # Given a label from `src_labels`, find the label of the CC from `dst_conncomp` that
    # had the most overlapping area with the corresponding CC in `src_conncomp` (if any
    # exists). If no label was found that satisfied the minimum overlap threshold,
    # returns `NO_OVERLAPPING_LABEL`.
    def get_max_overlapping_label(src_label: int) -> int:
        # Get a mask of pixels within the current CC.
        cc_mask = src_conncomp == src_label

        # Get the total area of the CC (i.e. the number of nonzero values in the mask).
        cc_area = np.count_nonzero(cc_mask)

        # Get the most frequent label from `dst_conncomp` within the masked region.
        dst_label, count = mode(dst_conncomp[cc_mask & dst_nonzero])

        # Check whether there was sufficient overlap between the two labels.
        if count >= min_overlap * cc_area:
            return dst_label
        else:
            return NO_OVERLAPPING_LABEL

    return {src_label: get_max_overlapping_label(src_label) for src_label in src_labels}


def relabel(
    conncomp: NDArray[np.unsignedinteger],
    label_mapping: Mapping[int, int],
) -> NDArray[np.unsignedinteger]:
    """
    Replace each label in `conncomp` with a new label from `label_mapping`.

    Given an array of provisional connected component labels `conncomp` and a mapping
    from provisional labels to final labels `label_mapping`, create a new array of
    connected component labels by replacing each provisional label with the
    corresponding final label.

    The set of unique nonzero labels in `conncomp` must be a subset of the keys of
    `label_mapping`.

    Zero-valued elements of `conncomp` are treated as masked out (i.e. not part of any
    connected component). They are not relabeled.

    Parameters
    ----------
    conncomp : numpy.ndarray
        The input array of provisional connected component labels. This array is not
        modified by the function.
    label_mapping : mapping
        Defines a mapping from each unique nonzero label in `conncomp` to the
        corresponding final label to assign to that component.

    Returns
    -------
    relabeled : numpy.ndarray
        A new array with the same shape and dtype as `conncomp` resulting from replacing
        each input connected component label with the corresponding label from
        `label_mapping`.
    """
    # Create the new connected components (CC) array, initially filled with zeros.
    new_conncomp = np.zeros_like(conncomp)

    # Loop over unique CC labels in the original `conncomp` array.
    for old_label in unique_nonzero_integers(conncomp):
        # Get a mask of pixels within the current CC.
        mask = conncomp == old_label

        # Get the corresponding final label.
        new_label = label_mapping[old_label]

        # Assign the new label to masked pixels in the output array.
        new_conncomp[mask] = new_label

    return new_conncomp


def extract_scalar(arr: np.ndarray) -> Any:
    """Extract the scalar value from an array containing a single element."""
    if arr.size != 1:
        raise ValueError(f"array size must be equal to 1, got {arr.size}")
    return np.squeeze(arr)[()]


def relabel_hires_conncomps(
    conncomp_hires: da.Array,
    conncomp_lores: da.Array,
    *,
    min_overlap: float = 0.5,
) -> da.Array:
    """
    Deduplicate and merge connected component labels resulting from tiled unwrapping.

    If a high-resolution interferogram is unwrapped by tiles, each tile may be assigned
    a set of connected component (CC) labels independently from the surrounding tiles.
    As a result, some CC labels may not be unique across tiles. Furthermore, if regions
    of reliable unwrapped phase spanned multiple tiles, they may be assigned different
    labels in different tiles.

    This function attempts to resolve these issues as a post-processing step by using a
    set of low-resolution CCs resulting from coarse unwrapping of the same
    interferogram. Unlike the high-resolution CCs, each low-resolution CC is assumed to
    be assigned a single unique label.

    For each high-resolution CC in each tile, the low-resolution CC that it shares the
    most overlapping area with is found. Then each CC is relabeled according to the
    low-resolution CC that it most overlapped with. If two or more high-resolution
    components share the same most-overlapping low-resolution CC, then they will be
    assigned the same label. High-resolution CCs that most overlapped with different
    low-resolution CCs will be assigned distinct labels. Each high-resolution CC that
    did not overlap with any low-resolution CC will be assigned a unique label.

    After relabeling, each unique connected component is assigned a new positive integer
    label in [1, 2, ..., N], where N is the total number of unique connected components.

    Zero-valued pixels are treated as masked out (i.e. not part of any connected
    component). They are not relabeled.

    Parameters
    ----------
    conncomp_hires : dask.array.Array
        The initial high-resolution connected components. A two-dimensional array of
        nonnegative integer values. Each chunk of the array is assumed to have been
        independently assigned its connected component labels, such that labels may not
        be unique across chunks and some components that span multiple chunks may have
        been assigned multiple labels.
    conncomp_lores : dask.array.Array
        An array of connected component labels resulting from coarse unwrapping. A
        two-dimensional array of nonnegative integer values with the same shape and
        chunk sizes as `conncomp_hires`. Unlike the high-resolution connected
        components, each connected component in `conncomp_lores` is assumed to be
        assigned a single unique label.
    min_overlap : float, optional
        Minimum intersection between components in order to be considered overlapping,
        as a fraction of the area of the high-resolution component area. Must be in the
        range (0, 1]. Defaults to 0.5.

    Returns
    -------
    new_conncomp_hires : dask.array.Array
        The array of updated high-resolution connected component labels.
    """
    # The high-res and low-res connected component (CC) arrays should each be 2-D arrays
    # with the same shape & chunk sizes.
    if conncomp_hires.ndim != 2:
        raise ValueError("the input connected components must be 2-D arrays")
    if conncomp_hires.shape != conncomp_lores.shape:
        raise ValueError(
            "shape mismatch: the high-res and low-res connected components arrays must"
            " have the same shape"
        )
    if conncomp_hires.chunks != conncomp_lores.chunks:
        raise ValueError(
            "the high-res and low-res connected components arrays must have the same"
            " chunk sizes"
        )

    # For each high-res CC in each tile, find the label of the low-res CC that most
    # overlapped with it, if any such component exists. The result is an array with
    # shape equal to `conncomp_hires.numblocks` of dicts mapping from high-res labels to
    # the corresponding low-res labels (one dict per tile).
    label_mappings = da.map_blocks(
        lambda cc_hires, cc_lores, min_overlap: np.atleast_2d(
            find_max_overlapping_labels(cc_hires, cc_lores, min_overlap=min_overlap)
        ),
        conncomp_hires,
        conncomp_lores,
        min_overlap=min_overlap,
        meta=np.empty((), dtype=np.object_),
    ).compute()

    # Get the set of all low-res CC labels from among all tiles that overlapped with any
    # high-res CC. This is the set of all unique values (not keys) from among dicts in
    # `label_mappings`.
    mapped_labels = get_all_unique_values(label_mappings.flat)

    # An inexhaustible generator that yields new unique positive-valued connected
    # component labels not found in the set of existing low-res labels.
    new_unique_labels = (
        label for label in itertools.count(1) if label not in mapped_labels
    )

    # Update the label mappings to replace `NO_OVERLAPPING_LABEL` values with new unique
    # labels.
    for label_mapping in label_mappings.flat:
        for key, val in label_mapping.items():
            if val == NO_OVERLAPPING_LABEL:
                label_mapping[key] = next(new_unique_labels)

    # Once more, get the set of all mapped-to labels in `label_mappings` after we
    # finished updating it to replace `NO_OVERLAPPING_LABEL` values with new labels.
    updated_mapped_labels = get_all_unique_values(label_mappings.flat)

    # We would like the final set of connected component labels to be the set of natural
    # numbers 1 through N, where N is the total number of connected components.
    # Currently, that's not necessarily the case in `updated_mapped_labels` -- due to
    # merging of equivalent labels, there could be some "gaps" in the natural sequence
    # of labels. So we define an additional mapping from `updated_mapped_labels` to the
    # set of final labels, which will be the natural numbers 1 through N.
    final_labels = dict(zip(updated_mapped_labels, itertools.count(1)))

    # Create a new array of label mappings, with one dict per tile in the original
    # high-res CC array. Each dict defines a mapping from the original high-res
    # labels to the corresponding final labels for each CC in the tile.
    final_label_mappings = label_mappings.copy()
    for label_mapping in final_label_mappings.flat:
        for key, val in label_mapping.items():
            label_mapping[key] = final_labels[val]

    # Break the `final_label_mappings` array up into chunks (one chunk per tile in the
    # input `conncomp_hires` array).
    final_label_mappings = da.from_array(final_label_mappings, chunks=(1, 1))
    assert final_label_mappings.numblocks == conncomp_hires.numblocks

    # Finally, create the output array of updated connected component labels by
    # replacing each high-res connected component label with the new corresponding label
    # from `final_label_mappings`.
    return da.map_blocks(
        lambda conncomp, label_mapping: relabel(
            conncomp, extract_scalar(label_mapping)
        ),
        conncomp_hires,
        final_label_mappings,
        meta=conncomp_hires._meta,
    )
