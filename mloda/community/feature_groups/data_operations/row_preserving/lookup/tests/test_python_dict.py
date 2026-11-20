"""Tests for the PythonDictLookup Implementation."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.community.feature_groups.data_operations.row_preserving.lookup.base import LookupUnmatchedError
from mloda.community.feature_groups.data_operations.row_preserving.lookup.python_dict_lookup import (
    PythonDictLookup,
)
from mloda.community.feature_groups.data_operations.row_preserving.lookup.tests.test_base import (
    _EmptyMappingHookLookup,
    _HookLookup,
)
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, extract_column, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.lookup.lookup import (
    LookupTestBase,
)


class TestPythonDictLookup(PythonDictTestMixin, LookupTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictLookup

    @classmethod
    def typed_null_source_becomes_untyped(cls) -> bool:
        return True

    @classmethod
    def hook_group_class(cls) -> Any:
        return _HookLookup

    @classmethod
    def invalid_hook_group_class(cls) -> Any:
        return _EmptyMappingHookLookup

    def test_out_of_range_int_in_column_rejected(self) -> None:
        fs = make_feature_set("v__lookup", lookup_mapping={1: 10}, on_unmatched="keep")
        with pytest.raises(ValueError, match="'v'") as excinfo:
            calculate_with(PythonDictLookup, {"v": [1, 2**70, None]}, fs)
        assert "outside int64" in str(excinfo.value)
        assert not isinstance(excinfo.value, LookupUnmatchedError)

    def test_in_range_uint64_values_serve_int_keys(self) -> None:
        """Rows carry no dtype, so a UINT64 column with in-range values maps; the reference rejects uint64."""
        fs = make_feature_set("v__lookup", lookup_mapping={1: 10, 2: 20}, on_unmatched="null")
        result = calculate_with(PythonDictLookup, {"v": [1, 2, None]}, fs)
        assert extract_column(result, "v__lookup") == [10, 20, None]
