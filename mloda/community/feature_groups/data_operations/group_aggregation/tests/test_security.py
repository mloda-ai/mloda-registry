"""Security tests for group aggregation."""

from __future__ import annotations

from typing import Any

from mloda.testing.feature_groups.data_operations.security import SecurityTestBase

from mloda.community.feature_groups.data_operations.group_aggregation.base import (
    GroupAggregationFeatureGroup,
)


class TestGroupAggregationSecurity(SecurityTestBase):
    @classmethod
    def feature_group_class(cls) -> Any:
        return GroupAggregationFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return set(GroupAggregationFeatureGroup.AGGREGATION_TYPES)

    @classmethod
    def config_key(cls) -> str:
        return "aggregation_type"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"value_int__{operation}_grouped"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        return {"in_features": "value_int", "partition_by": ["region"]}
