"""Security tests for rank operations."""

from __future__ import annotations

from typing import Any

from mloda.testing.feature_groups.data_operations.security import SecurityTestBase

from mloda.community.feature_groups.data_operations.row_preserving.rank.base import (
    RankFeatureGroup,
)


class TestRankSecurity(SecurityTestBase):
    @classmethod
    def feature_group_class(cls) -> Any:
        return RankFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return set(RankFeatureGroup.RANK_TYPES)

    @classmethod
    def config_key(cls) -> str:
        return "rank_type"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"value_int__{operation}_ranked"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        return {"in_features": "value_int", "partition_by": ["region"], "order_by": "value_int"}
