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
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
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
