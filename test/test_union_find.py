from collections.abc import Iterable

import pytest

import tophu


def count(iterable: Iterable) -> int:
    """Count the number of items in an iterable."""
    return sum(1 for _ in iterable)


class TestDisjointSetForest:
    def test_default_init(self):
        # Check that a default-constructed `DisjointSetForest` is empty.
        forest = tophu.DisjointSetForest()
        items = list(forest.items())
        assert len(items) == 0

    def test_init(self):
        # Construct a disjoint-set forest.
        items = {"a", "b", "c", "d"}
        forest = tophu.DisjointSetForest(items)

        # Check the contents of the forest.
        assert set(forest.items()) == items

        # Check that each item is its own disjoint subset.
        for item in forest.items():
            assert forest.find(item) == item

    def test_init_duplicate_item(self):
        # Check that `DisjointSetForest` only stores unique items upon construction.
        # Duplicate items are discarded.
        items = ["a", "b", "c", "b"]
        forest = tophu.DisjointSetForest(items)
        assert forest.num_items() == 3
        assert list(forest.items()) == ["a", "b", "c"]

    def test_contains(self):
        # Construct a disjoint-set forest.
        items = {"a", "b", "c", "d"}
        forest = tophu.DisjointSetForest(items)

        # Test `contains()` method.
        assert forest.contains("b")
        assert not forest.contains("e")

        # Test `in` and `not in` operators.
        assert "b" in forest
        assert "e" not in forest

    def test_add_item(self):
        forest = tophu.DisjointSetForest()
        assert "a" not in forest

        forest.add_item("a")
        assert "a" in forest

        assert forest.find("a") == "a"

    def test_add_duplicate_item(self):
        # Construct a disjoint-set forest and add some items to it.
        forest = tophu.DisjointSetForest()
        forest.add_item("a")
        forest.add_item("b")

        assert forest.num_items() == 2
        assert forest.get_parent("a") == "a"
        assert forest.get_parent("b") == "b"

        # Check that adding a duplicate item has no effect.
        forest.add_item("a")
        assert forest.num_items() == 2
        assert forest.get_parent("a") == "a"
        assert forest.get_parent("b") == "b"

        # Adding a duplicate of an existing item should not affect the parent of the
        # existing item.
        forest.set_parent("a", "b")
        forest.add_item("a")
        assert forest.get_parent("a") == "b"

    def test_add_items(self):
        # Construct an empty `DisjointSetForest`.
        forest = tophu.DisjointSetForest()

        # Add items.
        items = {"a", "b", "c", "d"}
        forest.add_items(items)

        # Check the contents of the forest.
        assert set(forest.items()) == items

        # Check that each item is its own disjoint subset.
        for item in forest.items():
            assert forest.find(item) == item

    def test_iter(self):
        # Construct a disjoint-set forest and add some items to it.
        forest = tophu.DisjointSetForest(["a", "b", "c"])
        forest.add_item("d")
        forest.add_item("e")

        # Test iterating over the forest. The iteration order should match the order in
        # which items were added to the forest.
        items = [item for item in forest]
        assert items == ["a", "b", "c", "d", "e"]

        # Test the `items()` method.
        items = list(forest.items())
        assert items == ["a", "b", "c", "d", "e"]

    def test_roots(self):
        # Construct a disjoint-set forest.
        items = {"a", "b", "c", "d"}
        forest = tophu.DisjointSetForest(items)

        # Initially, each node in the forest is a root node.
        assert forest.roots() == items

        # Test the `roots()` method after growing trees.
        forest.set_parent("a", "b")
        forest.set_parent("c", "d")
        assert forest.roots() == {"b", "d"}

        forest.set_parent("b", "d")
        assert forest.roots() == {"d"}

    def test_len(self):
        forest = tophu.DisjointSetForest()
        assert len(forest) == 0

        items = {"a", "b", "c", "d"}
        forest = tophu.DisjointSetForest(items)
        assert len(forest) == 4

        forest.add_item("e")
        assert len(forest) == 5

    def test_num_items(self):
        forest = tophu.DisjointSetForest()
        assert forest.num_items() == 0

        items = {"a", "b", "c", "d"}
        forest = tophu.DisjointSetForest(items)
        assert forest.num_items() == 4

        forest.add_item("e")
        assert forest.num_items() == 5

    def test_num_trees(self):
        # Construct a disjoint-set forest.
        items = {"a", "b", "c", "d"}
        forest = tophu.DisjointSetForest(items)

        # Initially, each node in the forest is a singleton tree.
        assert forest.num_trees() == forest.num_items()

        # Merge some trees.
        forest.union("a", "b")
        forest.union("c", "d")
        assert forest.num_trees() == 2

        forest.union("b", "d")
        assert forest.num_trees() == 1

    def test_eq(self):
        forest1 = tophu.DisjointSetForest({"a", "b", "c"})
        forest2 = tophu.DisjointSetForest({"a", "b", "c"})
        forest3 = tophu.DisjointSetForest({"A", "B", "C"})

        # Test equality comparison.
        assert forest1 == forest2
        assert forest1 != forest3

        # Test equality comparison against other types.
        assert forest1 != "asdf"

        # Equality comparison should be sensitive to the topology of the underlying
        # graph.
        forest2.set_parent("a", "b")
        assert forest1 != forest2

    def test_copy(self):
        # Construct a disjoint-set forest.
        items = {"a", "b", "c"}
        forest = tophu.DisjointSetForest(items)

        # Create a copy.
        copy = forest.copy()

        # The copy should be equivalent to the original.
        assert copy == forest

        # Modifying the copy should not affect the original.
        copy.set_parent("a", "b")
        assert forest.get_parent("a") != "b"
        assert copy != forest

    def test_get_parent(self):
        # Construct a disjoint-set forest.
        items = {"a", "b"}
        forest = tophu.DisjointSetForest(items)

        # Initially, each item should be a root node (its parent node is itself).
        assert forest.get_parent("a") == "a"
        assert forest.get_parent("b") == "b"

        # After setting a new parent node, `get_parent()` should return the new parent.
        forest.set_parent("a", "b")
        assert forest.get_parent("a") == "b"
        assert forest.get_parent("b") == "b"

    def test_bad_get_parent(self):
        # Construct a disjoint-set forest.
        items = {"a", "b", "c", "d"}
        forest = tophu.DisjointSetForest(items)

        # Check that `get_parent()` fails if the input item is not part of the set.
        with pytest.raises(ValueError, match="item not found: e"):
            forest.get_parent("e")

    def test_bad_set_parent(self):
        # Construct a disjoint-set forest.
        items = {"a", "b", "c", "d"}
        forest = tophu.DisjointSetForest(items)

        # Check that `set_parent()` fails if either the parent or child was not already
        # part of the set.
        with pytest.raises(ValueError, match="item not found: e"):
            forest.set_parent("e", "a")
        with pytest.raises(ValueError, match="item not found: e"):
            forest.set_parent("a", "e")

    def test_iter_parents(self):
        # Construct a disjoint-set forest.
        items = {"a", "b", "c", "d"}
        forest = tophu.DisjointSetForest(items)

        # Form a single linear tree by assigning parents.
        forest.set_parent("a", "b")
        forest.set_parent("b", "c")
        forest.set_parent("c", "d")

        # Check each node's ancestors.
        assert list(forest.iter_parents("a")) == ["b", "c", "d"]
        assert list(forest.iter_parents("b")) == ["c", "d"]
        assert list(forest.iter_parents("c")) == ["d"]
        assert list(forest.iter_parents("d")) == []

    def test_find(self):
        # Construct a disjoint-set forest.
        items = {"a", "b", "c", "d", "e", "f"}
        forest = tophu.DisjointSetForest(items)

        # Form a tree containing {'a', 'b', 'c', 'd'} (with 'a' and 'c' as sister
        # nodes), and another tree containing {'e', 'f'}.
        forest.set_parent("a", "b")
        forest.set_parent("c", "b")
        forest.set_parent("b", "d")
        forest.set_parent("e", "f")

        # Test the `find()` method.
        for item in ["a", "b", "c", "d"]:
            assert forest.find(item) == "d"
        for item in ["e", "f"]:
            assert forest.find(item) == "f"

    def test_union(self):
        # Construct a disjoint-set forest.
        items = {"a", "b", "c", "d", "e"}
        forest = tophu.DisjointSetForest(items)

        # Check that 'a' and 'b' belong to disjoint subsets before the `union()`
        # operation but not afterwards.
        assert forest.is_disjoint("a", "b")
        forest.union("a", "b")
        assert not forest.is_disjoint("a", "b")

        # Merge the subsets containing 'c' and 'd'.
        forest.union("c", "d")

        # Currently, {'a', 'b'} and {'c', 'd'} are two disjoint subsets. Check that 'a'
        # and 'd' belong to disjoint subsets before invoking `union('b', 'c')`, but not
        # afterwards.
        assert forest.is_disjoint("a", "d")
        forest.union("b", "c")
        assert not forest.is_disjoint("a", "d")

        # Each item in the forest except for 'e' should now have the same root node.
        root = forest.find("a")
        for item in ["b", "c", "d"]:
            assert forest.find(item) == root
        assert forest.find("e") != root

    def test_union_noop(self):
        # Construct a disjoint-set forest.
        items = {"a", "b", "c"}
        forest = tophu.DisjointSetForest(items)
        forest.union("a", "b")
        forest.union("b", "c")

        # Check that `union()` has no effect if the two subsets were already merged.
        copy = forest.copy()
        forest.union("a", "c")
        assert forest == copy

    def test_flatten(self):
        # Construct a disjoint-set forest.
        items = {"a", "b", "c", "d", "e", "f"}
        forest = tophu.DisjointSetForest(items)

        # Form a tree containing {'a', 'b', 'c', 'd'} (with 'a' and 'c' as sister
        # nodes), and another tree containing {'e', 'f'}.
        forest.set_parent("a", "b")
        forest.set_parent("c", "b")
        forest.set_parent("b", "d")
        forest.set_parent("e", "f")

        # Check that flattening does not affect the result of the `find()` operation for
        # each node in the forest.
        old_roots = [forest.find(item) for item in forest.items()]
        forest.flatten()
        new_roots = [forest.find(item) for item in forest.items()]
        assert old_roots == new_roots

        # After flattening, each node's parent should be the root of the tree to which
        # it belongs.
        for item in forest.items():
            assert forest.get_parent(item) == forest.find(item)

        # Get the number of ancestors of a node in the forest (i.e. its depth within the
        # corresponding tree).
        def num_ancestors(item):
            ancestors = forest.iter_parents(item)
            return count(ancestors)

        # After flattening, each node in the forest should have at most 1 ancestor.
        for item in forest.items():
            assert num_ancestors(item) <= 1
