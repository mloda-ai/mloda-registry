"""Tests for PolarsLazyLookup compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("polars")

import polars as pl

from mloda.community.feature_groups.data_operations.row_preserving.lookup.base import LookupUnmatchedError
from mloda.community.feature_groups.data_operations.row_preserving.lookup.polars_lazy_lookup import (
    PolarsLazyLookup,
)
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.polars_lazy import PolarsLazyTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.lookup.lookup import (
    EmptyMappingHook,
    LookupTestBase,
    NutsMappingHook,
)


class PolarsLazyNutsLookup(NutsMappingHook, PolarsLazyLookup):
    """Hook subclass: matches ``*__nuts2024`` by name and resolves its own mapping."""


class _EmptyMappingLookup(EmptyMappingHook, PolarsLazyLookup):
    """Hook subclass whose resolved mapping is empty (invalid)."""


class TestPolarsLazyLookup(PolarsLazyTestMixin, LookupTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyLookup

    @classmethod
    def hook_implementation_class(cls) -> Any:
        return PolarsLazyNutsLookup

    @classmethod
    def invalid_hook_implementation_class(cls) -> Any:
        return _EmptyMappingLookup

    def test_source_column_named_like_helper_column_raises_unmatched(self) -> None:
        fs = make_feature_set("__mloda_n__lookup", lookup_mapping={"a": "x"}, on_unmatched="raise")
        with pytest.raises(LookupUnmatchedError):
            PolarsLazyLookup.calculate_feature(pl.LazyFrame({"__mloda_n": ["a", "zzz"]}), fs).collect()

    @pytest.mark.parametrize("dtype", [pl.Int128, pl.UInt128], ids=["int128", "uint128"])
    @pytest.mark.parametrize("policy", ["null", "raise"])
    def test_128_bit_int_column_rejected_as_key(self, dtype: Any, policy: str) -> None:
        fs = make_feature_set("v__lookup", lookup_mapping={1: 10}, on_unmatched=policy)
        frame = pl.LazyFrame({"v": pl.Series("v", [1, 2], dtype=dtype)})
        with pytest.raises(ValueError, match="'v'") as excinfo:
            PolarsLazyLookup.calculate_feature(frame, fs).collect()
        assert not isinstance(excinfo.value, LookupUnmatchedError)
