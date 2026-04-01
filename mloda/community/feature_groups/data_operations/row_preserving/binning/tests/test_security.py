"""Security tests for binning operations."""

from __future__ import annotations

from typing import Any

from mloda.testing.feature_groups.data_operations.security import SecurityTestBase

from mloda.community.feature_groups.data_operations.row_preserving.binning.base import (
    BINNING_OPS,
    BinningFeatureGroup,
)


class TestBinningSecurity(SecurityTestBase):
    @classmethod
    def feature_group_class(cls) -> Any:
        return BinningFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return set(BINNING_OPS)

    @classmethod
    def config_key(cls) -> str:
        return "binning_op"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"value_int__{operation}_5"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        return {"in_features": "value_int", "n_bins": 5}
