import pytest

import tophu


class TestUnwrapCallback:
    def test_abstract_class(self):
        # Check that `UnwrapCallback` is an abstract class -- it cannot be directly
        # instantiated.
        with pytest.raises(TypeError, match="Protocols cannot be instantiated"):
            tophu.UnwrapCallback()


class TestSnaphuUnwrapper:
    def test_interface(self):
        # Check that `SnaphuUnwrapper` satisfies the interface requirements of
        # `UnwrapCallback`.
        assert issubclass(tophu.SnaphuUnwrap, tophu.UnwrapCallback)
