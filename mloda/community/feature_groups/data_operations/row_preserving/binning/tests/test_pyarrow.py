"""Tests for PyArrowBinning compute implementation."""

from __future__ import annotations

import warnings
from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.binning.pyarrow_binning import (
    PyArrowBinning,
)
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.pyarrow import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.binning.binning import (
    EXPECTED_QBIN_3,
    BinningTestBase,
)


class TestPyArrowBinning(PyArrowTestMixin, BinningTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowBinning

    def test_qbin_no_future_warning_with_nulls(self) -> None:
        """qbin with a null value present must not raise pyarrow's null_placement FutureWarning."""
        fs = make_feature_set("value_int__qbin_3")
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__qbin_3")
        assert result_col == EXPECTED_QBIN_3
