import pytest

import tophu


class TestTiledPartition:
    @pytest.fixture
    def tiles2d(self) -> tophu.TiledPartition:
        return tophu.TiledPartition(shape=(100, 101), ntiles=(4, 3))

    @pytest.fixture
    def overlapped(self) -> tophu.TiledPartition:
        return tophu.TiledPartition(shape=1000, ntiles=10, overlap=50)

    @pytest.fixture
    def snapped(self) -> tophu.TiledPartition:
        return tophu.TiledPartition(
            shape=(30, 40, 50),
            ntiles=(3, 4, 5),
            snap_to=(5, 6, 7),
        )

    def test_ntiles(self, tiles2d, overlapped, snapped):
        assert tiles2d.ntiles == (4, 3)
        assert overlapped.ntiles == (10,)
        assert snapped.ntiles == (3, 4, 5)

    def test_tiledims(self, tiles2d, overlapped, snapped):
        assert tiles2d.tiledims == (25, 34)
        assert overlapped.tiledims == (150,)
        assert snapped.tiledims == (10, 12, 14)

    def test_overlap(self, tiles2d, overlapped, snapped):
        assert tiles2d.overlap == (0, 0)
        assert overlapped.overlap == (50,)
        assert snapped.overlap == (0, 2, 4)

    def test_strides(self, tiles2d, overlapped, snapped):
        assert tiles2d.strides == tiles2d.tiledims
        assert overlapped.strides == (100,)
        assert snapped.strides == (10, 10, 10)

    def test_getitem(self, tiles2d, overlapped, snapped):
        assert tiles2d[0, 0] == (slice(0, 25), slice(0, 34))
        assert tiles2d[-1, -1] == (slice(75, 100), slice(68, 101))
        assert overlapped[5] == (slice(500, 650),)
        assert snapped[1, 2, 3] == (slice(10, 20), slice(20, 32), slice(30, 44))

    def test_iter(self, tiles2d):
        assert list(tiles2d) == [
            (slice(0, 25), slice(0, 34)),
            (slice(0, 25), slice(34, 68)),
            (slice(0, 25), slice(68, 101)),
            (slice(25, 50), slice(0, 34)),
            (slice(25, 50), slice(34, 68)),
            (slice(25, 50), slice(68, 101)),
            (slice(50, 75), slice(0, 34)),
            (slice(50, 75), slice(34, 68)),
            (slice(50, 75), slice(68, 101)),
            (slice(75, 100), slice(0, 34)),
            (slice(75, 100), slice(34, 68)),
            (slice(75, 100), slice(68, 101)),
        ]

    def test_ntiles_length_mismatch(self):
        errmsg = "size mismatch: shape and ntiles must have same length"
        with pytest.raises(ValueError, match=errmsg):
            tophu.TiledPartition(shape=(3, 4, 5), ntiles=(1, 2))

    def test_bad_shape(self):
        with pytest.raises(ValueError, match="array dimensions must be > 0"):
            tophu.TiledPartition(shape=(3, 0, 5), ntiles=(1, 2, 1))

    def test_bad_ntiles(self):
        with pytest.raises(ValueError, match="number of tiles must be > 0"):
            tophu.TiledPartition(shape=(3, 4, 5), ntiles=(1, 0, 1))

    def test_overlap_length_mismatch(self):
        errmsg = "size mismatch: shape and overlap must have same length"
        with pytest.raises(ValueError, match=errmsg):
            tophu.TiledPartition(shape=(3, 4, 5), ntiles=(1, 2, 1), overlap=(2, 1))

    def test_bad_overlap(self):
        with pytest.raises(ValueError, match="overlap between tiles must be >= 0"):
            tophu.TiledPartition(shape=(3, 4, 5), ntiles=(1, 2, 1), overlap=(0, -1, 1))

    def test_snap_to_length_mismatch(self):
        errmsg = "size mismatch: shape and snap_to must have same length"
        with pytest.raises(ValueError, match=errmsg):
            tophu.TiledPartition(shape=(3, 4, 5), ntiles=(1, 2, 1), snap_to=(4, 4))

    def test_bad_snap_to(self):
        with pytest.raises(ValueError, match="snap_to must be > 0"):
            tophu.TiledPartition(shape=(3, 4, 5), ntiles=(1, 2, 1), snap_to=(4, 0, 5))
