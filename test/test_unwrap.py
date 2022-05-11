import pytest

import tophu


class TestUnwrapFunc:
    def test_abstract_class(self):
        # Check that `UnwrapFunc` is an abstract class -- it cannot be directly
        # instantiated.
        with pytest.raises(TypeError, match="Protocols cannot be instantiated"):
            tophu.UnwrapFunc()


class TestSnaphuUnwrapper:
    def test_interface(self):
        # Check that `SnaphuUnwrapper` satisfies the interface requirements of
        # `UnwrapFunc`.
        assert issubclass(tophu.SnaphuUnwrapper, tophu.UnwrapFunc)
