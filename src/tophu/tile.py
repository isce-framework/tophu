import itertools
from typing import Iterable, Iterator, Optional, SupportsInt, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "TiledPartition",
]


IntOrInts = Union[SupportsInt, Iterable[SupportsInt]]
NDSlice = Tuple[slice, ...]


def as_tuple_of_int(ints: IntOrInts) -> Tuple[int, ...]:
    """
    Convert the input to a tuple of ints.

    Parameters
    ----------
    ints : int or iterable of int
        One or more integers.

    Returns
    -------
    out : tuple of int
        Tuple containing the inputs.
    """
    try:
        return (int(ints),)  # type: ignore
    except TypeError:
        return tuple([int(i) for i in ints])  # type: ignore


def ceil_divide(n: ArrayLike, d: ArrayLike) -> NDArray:
    """
    Return the smallest integer greater than or equal to the quotient of the inputs.

    Computes integer division of dividend `n` by divisor `d`, rounding up instead of
    truncating.

    Parameters
    ----------
    n : array_like
        Numerator.
    d : array_like
        Denominator.

    Returns
    -------
    q : numpy.ndarray
        Quotient.
    """
    n = np.asanyarray(n)
    d = np.asanyarray(d)
    return (n + d - np.sign(d)) // d


def round_up_to_next_multiple(n: ArrayLike, base: ArrayLike) -> NDArray:
    """Round up to the next smallest multiple of `base`."""
    n = np.asanyarray(n)
    base = np.asanyarray(base)

    # Determine output datatype based on the inputs using NumPy's type promotion rules.
    # In particular, if both inputs were integer-valued, the result should also be
    # integer-valued.
    out_dtype = np.result_type(n, base)

    out = base * np.ceil(n / base)
    return np.require(out, dtype=out_dtype)


class TiledPartition:
    """
    A partitioning of an N-dimensional array into tiles.

    A `TiledPartition` object subdivides an array of given shape into one or more
    roughly equally sized, possibly overlapping, tiles. Each tile corresponds to a block
    of data from the original array (or another array of similar shape). The full set of
    tiles spans the array.

    A tile is represented by an index expression (i.e. a tuple of `slice` objects) that
    can be used to access the corresponding block of data from the partitioned array.
    Individual tiles can be accessed using the subscripting operator. In addition,
    `TiledPartition` objects support iterating over the set of tiles in arbitrary order.
    """

    def __init__(
        self,
        shape: IntOrInts,
        ntiles: IntOrInts,
        overlap: Optional[IntOrInts] = None,
        snap_to: Optional[IntOrInts] = None,
    ):
        """
        Construct a new `TiledPartition` object.

        Subdivides an array of shape `shape` into (possibly overlapping) tiles of
        roughly equal size.

        Parameters
        ----------
        shape : int or iterable of int
            Shape of the array to be partitioned into tiles.
        ntiles : int or iterable of int
            Number of tiles along each array axis. Must be the same length as `shape`.
        overlap : int or iterable of int or None, optional
            Requested overlap along each array axis, in samples, between adjacent tiles.
            The actual overlap may be larger if `snap_to` is also specified.
            (default: None)
        snap_to : int or iterable of int or None, optional
            If specified, force tile dimensions to be a multiple of this value.
            (default: None)
        """
        # Convert `shape` and `ntiles` into tuples of ints.
        shape = as_tuple_of_int(shape)
        ntiles = as_tuple_of_int(ntiles)

        # Number of dimensions of the partitioned array.
        ndim = len(shape)

        if len(ntiles) != ndim:
            raise ValueError("size mismatch: shape and ntiles must have same length")
        if any(map(lambda s: s <= 0, shape)):
            raise ValueError("array dimensions must be > 0")
        if any(map(lambda n: n <= 0, ntiles)):
            raise ValueError("number of tiles must be > 0")

        # Get the stride length, in samples, between the start index of adjacent tiles
        # along each axis.
        strides = ceil_divide(shape, ntiles)

        # Normalize `overlap` to a tuple of ints.
        if overlap is None:
            overlap = ndim * (0,)
        else:
            overlap = as_tuple_of_int(overlap)
            if len(overlap) != ndim:
                raise ValueError(
                    "size mismatch: shape and overlap must have same length"
                )
            if any(map(lambda o: o < 0, overlap)):
                raise ValueError("overlap between tiles must be >= 0")

        # Normalize `snap_to` to a tuple of ints.
        if snap_to is None:
            snap_to = ndim * (1,)
        else:
            snap_to = as_tuple_of_int(snap_to)
            if len(snap_to) != ndim:
                raise ValueError(
                    "size mismatch: shape and snap_to must have same length"
                )
            if any(map(lambda s: s <= 0, snap_to)):
                raise ValueError("snap_to must be > 0")

        # Get the dimensions of each tile (except for the last tile along each axis).
        tiledims = round_up_to_next_multiple(strides + overlap, snap_to)

        # Tile dimensions should not exceed the full array dimensions.
        # XXX: how to handle the case where `shape` is not a multiple of `snap_to`?
        tiledims = np.minimum(tiledims, shape)

        self._shape = shape
        self._ntiles = ntiles
        self._strides = strides
        self._tiledims = tiledims

    @property
    def ntiles(self) -> Tuple[int, ...]:
        """tuple of int : Number of tiles along each array axis."""
        return self._ntiles

    @property
    def tiledims(self) -> Tuple[int, ...]:
        """
        tuple of int : Shape of a typical tile. The last tile along each axis may be
        smaller.
        """
        return tuple(self._tiledims)

    @property
    def strides(self) -> Tuple[int, ...]:
        """
        tuple of int : Step size between the start of adjacent tiles along each
        axis.
        """
        return tuple(self._strides)

    @property
    def overlap(self) -> Tuple[int, ...]:
        """tuple of int : Overlap between adjacent tiles along each axis."""
        return tuple(self._tiledims - self._strides)

    def __getitem__(self, index: IntOrInts) -> NDSlice:
        """
        Access a tile.

        Returns an index expression corresponding to a tile within the partitioned
        array. The resulting object can be used to access a block of data from the
        array.

        Parameters
        ----------
        index : int or tuple of int
            Index of the tile.

        Returns
        -------
        tile : tuple of slice
            A tuple of slices that can be used to access the corresponding block of
            data from an array.
        """
        index = as_tuple_of_int(index)

        if len(index) != len(self.ntiles):
            raise ValueError("...")

        # Wrap around negative indices.
        def wrap_index(i: int, n: int) -> int:
            return i + n if (i < 0) else i

        index = [wrap_index(i, n) for (i, n) in zip(index, self._ntiles)]

        start = self._strides * index
        stop = np.minimum(start + self._tiledims, self._shape)

        return tuple([slice(a, b) for (a, b) in zip(start, stop)])

    def __iter__(self) -> Iterator[NDSlice]:
        """
        Iterate over tiles in arbitrary order.

        Yields
        ------
        tile : tuple of slice
            A tuple of slices that can be used to access the corresponding block of
            data from an array.
        """
        subindices = (range(n) for n in self.ntiles)
        for index in itertools.product(*subindices):
            yield self[index]
