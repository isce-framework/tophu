from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any, Dict, Generic, Set, TypeVar

__all__ = [
    "DisjointSetForest",
]


T = TypeVar("T", bound=Hashable)


class DisjointSetForest(Generic[T]):
    """
    A Union-Find data structure.

    `DisjointSetForest` stores a set of items that are partitioned into disjoint
    subsets. Each subset is identified by a unique representative exemplar chosen from
    among items within the subset. The data structure supports operations for adding
    items to the set, merging two subsets, and finding set representatives.

    Conceptually, a `DisjointSetForest` is represented by a forest of `parent pointer
    trees <https://en.wikipedia.org/wiki/Parent_pointer_tree>`_, with each node in the
    forest corresponding to an item in the set, and each tree corresponding to a
    disjoint subset of the full set. Nodes are linked to their parent node within the
    same tree (except for root nodes, which are linked to themselves or a sentinel
    node). The root node of each tree acts as the unique representative for that subset.

    `DisjointSetForest` can only be used with hashable objects.
    """

    def __init__(self, items: Iterable[T] = ()):
        """
        Construct a new `DisjointSetForest` object.

        Initially, each item in the set is treated as a disjoint singleton subset
        (represented by a tree containing a single node).

        Parameters
        ----------
        items : iterable
            Items within the set.
        """
        # Internally, the forest is stored as an associative container which maps each
        # node to its parent node. Root nodes are mapped to themselves.
        self._parents: Dict[T, T] = {}

        # Add nodes to the forest.
        self.add_items(items)

    def __contains__(self, item: T) -> bool:
        return self.contains(item)

    def __iter__(self) -> Iterable[T]:
        return self.items()

    def __len__(self) -> int:
        return self.num_items()

    def __eq__(self, other: Any) -> bool:
        return (type(self) == type(other)) and (self._parents == other._parents)

    def copy(self) -> DisjointSetForest[T]:
        """
        Create a shallow copy of the `DisjointSetForest` object.

        Returns
        -------
        copy : DisjointSetForest
            A copy of the disjoint-set forest.
        """
        other: DisjointSetForest[T] = DisjointSetForest()
        other._parents = self._parents.copy()
        return other

    def contains(self, item: T) -> bool:
        """
        Check whether an item is a member of the set.

        Parameters
        ----------
        item : object
            The item to check for.

        Returns
        -------
        found : bool
            True if a node in the forest contained `item`, otherwise False.
        """
        return item in self._parents

    def add_item(self, item: T) -> None:
        """
        Add an item to the set.

        This has no effect if the item is already a member of the set. Otherwise, it
        creates a new disjoint subset containing a single item.

        Parameters
        ----------
        item : object
            The new item to add.
        """
        # Do nothing if the item is already found in the forest.
        if self.contains(item):
            return

        # Otherwise, add the item as a new root node.
        self._parents[item] = item

    def add_items(self, items: Iterable[T]) -> None:
        """
        Add items to the set.

        Each item is added to the forest as a disjoint singleton subset (represented by
        a tree containing a single node).

        Adding an item that is already a member of set has no effect.

        Parameters
        ----------
        items : iterable
            The new items to add.
        """
        for item in items:
            self.add_item(item)

    def items(self) -> Iterable[T]:
        """
        Iterate over items within the set.

        Each item is visited in the order in which it was added to the set.

        Yields
        ------
        item : object
            An item within the set.
        """
        yield from self._parents.keys()

    def roots(self) -> Set[T]:
        """
        Get subset representatives.

        Find the root node of each tree in the forest. Each rooted tree in the forest
        corresponds to a disjoint subset of the total set of items. The root node is
        treated as a representative exemplar of the corresponding subset.

        Returns
        -------
        roots : set
            The set of root nodes of each tree in the forest.
        """
        return {self.find(item) for item in self.items()}

    def num_items(self) -> int:
        """
        Get the number of items in the set.

        Returns
        -------
        n : int
            The total number of nodes in the forest.
        """
        return len(self._parents)

    def num_trees(self) -> int:
        """
        Get the number of disjoint subsets in the set.

        Returns
        -------
        n : int
            The number of trees in the forest.
        """
        return len(self.roots())

    def get_parent(self, item: T) -> T:
        """
        Get the parent of the specified item.

        Parameters
        ----------
        item : object
            An item within the set.

        Returns
        -------
        parent : object
            The item occupying the parent node within the same tree.

        Raises
        ------
        ValueError
            If `item` is not found in the forest.

        Notes
        -----
        A root node's parent is itself.
        """
        try:
            return self._parents[item]
        except KeyError:
            raise ValueError(f"item not found: {item}")

    def set_parent(self, item: T, parent: T) -> None:
        """
        Set the parent of the specified item.

        Both the child and parent must already be set members.

        Parameters
        ----------
        item : object
            The child item.
        parent : object
            The new parent item.

        Raises
        ------
        ValueError
            If either `item` or `parent` is not found in the forest.
        """
        if item not in self._parents:
            raise ValueError(f"item not found: {item}")
        if parent not in self._parents:
            raise ValueError(f"item not found: {parent}")

        self._parents[item] = parent

    def iter_parents(self, item: T) -> Iterable[T]:
        """
        Iterate over ancestors of a specified node.

        Parameters
        ----------
        item : object
            An item within the forest.

        Yields
        ------
        parent : object
            An ancestor of the specified item.
        """
        child, parent = item, self.get_parent(item)
        while parent != child:
            yield parent
            child, parent = parent, self.get_parent(parent)

    def find(self, item: T) -> T:
        """
        Find the representative item of a subset.

        Parameters
        ----------
        item : object
            Provisional label.

        Returns
        -------
        root : object
            Representative for the set that contains `item`.

        Notes
        -----
        The subset representative is the root of the tree that contains `item`.
        """
        child, parent = item, self.get_parent(item)
        while parent != child:
            child, parent = parent, self.get_parent(parent)
        return parent

    def union(self, x: T, y: T, /) -> None:
        """
        Merge two subsets.

        Has no effect if the two items are already members of the same subset.

        Parameters
        ----------
        x, y : object
            Items to merge.
        """
        # Get the root of each node's tree.
        xroot = self.find(x)
        yroot = self.find(y)

        # If `x` and `y` are already members of the same tree, do nothing.
        if xroot == yroot:
            return

        # Merge the two trees.
        self.set_parent(xroot, yroot)

    def flatten(self) -> None:
        """
        Flatten the forest.

        Modify the forest's topology so that each node is a direct child of its root
        node.
        """
        for item in self.items():
            root = self.find(item)
            self.set_parent(item, root)

    def is_disjoint(self, x: T, y: T, /) -> bool:
        """
        Check whether two members of the set belong to disjoint subsets.

        Parameters
        ----------
        x, y : object
            Items within the set.

        Returns
        -------
        b : bool
            True if `x` and `y` belong to different trees, otherwise False.
        """
        xroot = self.find(x)
        yroot = self.find(y)
        return xroot != yroot
