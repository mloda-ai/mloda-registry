"""Tests for PandasLookup compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

import pandas as pd  # noqa: E402

from mloda.community.feature_groups.data_operations.row_preserving.lookup.pandas_lookup import (
    PandasLookup,
)
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.pandas import (
    PandasNullableIntTestMixin,
    PandasTestMixin,
)
from mloda.testing.feature_groups.data_operations.row_preserving.lookup.lookup import (
    EmptyMappingHook,
    LookupTestBase,
    NutsMappingHook,
)


class PandasNutsLookup(NutsMappingHook, PandasLookup):
    """Hook subclass: matches ``*__nuts2024`` by name and resolves its own mapping."""


class _EmptyMappingLookup(EmptyMappingHook, PandasLookup):
    """Hook subclass whose resolved mapping is empty (invalid)."""


class TestPandasLookup(PandasTestMixin, LookupTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PandasLookup

    @classmethod
    def typed_null_source_becomes_untyped(cls) -> bool:
        return True

    @classmethod
    def int_with_nulls_becomes_float(cls) -> bool:
        return True

    @classmethod
    def hook_implementation_class(cls) -> Any:
        return PandasNutsLookup

    @classmethod
    def invalid_hook_implementation_class(cls) -> Any:
        return _EmptyMappingLookup


class TestPandasLookupNullableInt(PandasNullableIntTestMixin, TestPandasLookup):
    """Runs the lookup suite on nullable pandas int columns."""

    @classmethod
    def typed_null_source_becomes_untyped(cls) -> bool:
        return False

    @classmethod
    def int_with_nulls_becomes_float(cls) -> bool:
        return False


class TestPandasDivergences:
    """Pins for pandas behavior documented in known-divergences.md."""

    def test_all_null_float64_column_skips_key_family_check(self) -> None:
        fs = make_feature_set("v__lookup", lookup_mapping={"a": "x"}, on_unmatched="null")
        result = PandasLookup.calculate_feature(pd.DataFrame({"v": [float("nan")] * 2}), fs)
        assert result["v__lookup"].isna().tolist() == [True, True]

    def test_float64_with_integral_values_accepted_as_int_keys(self) -> None:
        fs = make_feature_set("v__lookup", lookup_mapping={1: "one", 2: "two"}, on_unmatched="null")
        result = PandasLookup.calculate_feature(pd.DataFrame({"v": [1.0, 2.0, None, 3.0]}), fs)
        assert result["v__lookup"].tolist()[:2] == ["one", "two"]
        assert result["v__lookup"].isna().tolist()[2:] == [True, True]

    def test_non_integral_float64_rejected_for_int_keys(self) -> None:
        fs = make_feature_set("v__lookup", lookup_mapping={1: "one"}, on_unmatched="null")
        with pytest.raises(ValueError, match="'v'"):
            PandasLookup.calculate_feature(pd.DataFrame({"v": [1.0, 2.5]}), fs)

    def test_float64_with_inf_not_classed_int(self) -> None:
        fs = make_feature_set("v__lookup", lookup_mapping={1: "one"}, on_unmatched="null")
        with pytest.raises(ValueError, match="'v'"):
            PandasLookup.calculate_feature(pd.DataFrame({"v": [float("inf"), 1.0]}), fs)

    def test_object_column_with_out_of_range_int_rejected(self) -> None:
        fs = make_feature_set("v__lookup", lookup_mapping={1: 10}, on_unmatched="null")
        with pytest.raises(ValueError, match="'v'"):
            PandasLookup.calculate_feature(pd.DataFrame({"v": pd.Series([1, 2**70, None], dtype=object)}), fs)

    def test_object_int_column_with_null_maps_large_keys_exactly(self) -> None:
        fs = make_feature_set("v__lookup", lookup_mapping={2**60: "lo", 2**60 + 1: "hi"}, on_unmatched="null")
        source = pd.DataFrame({"v": pd.Series([2**60, 2**60 + 1, None], dtype=object)})
        result = PandasLookup.calculate_feature(source, fs)
        values = result["v__lookup"].tolist()
        assert values[:2] == ["lo", "hi"]
        assert pd.isna(values[2])

    @pytest.mark.parametrize(
        ("values", "mapping", "expected_dtype"),
        [
            pytest.param(["a", "b", None], {"a": "x", "b": "y"}, "object", id="str"),
            pytest.param(["a", "b", None], {"a": 1, "b": 2}, "Int64", id="int"),
            pytest.param(["a", "b", None], {"a": 1.5, "b": 2.5}, "float64", id="float"),
            pytest.param(["a", "b", None], {"a": True, "b": False}, "boolean", id="bool"),
        ],
    )
    def test_output_dtype_pinned(self, values: list[Any], mapping: dict[Any, Any], expected_dtype: str) -> None:
        fs = make_feature_set("v__lookup", lookup_mapping=mapping, on_unmatched="null")
        result = PandasLookup.calculate_feature(pd.DataFrame({"v": values}), fs)
        assert str(result["v__lookup"].dtype) == expected_dtype
