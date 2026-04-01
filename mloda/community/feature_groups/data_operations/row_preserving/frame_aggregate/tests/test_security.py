"""Security tests for frame aggregate operations."""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.security import SecurityTestBase

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
    FrameAggregateFeatureGroup,
    _AGGREGATION_TYPES,
)


class TestFrameAggregateSecurity(SecurityTestBase):
    @classmethod
    def feature_group_class(cls) -> Any:
        return FrameAggregateFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return set(_AGGREGATION_TYPES)

    @classmethod
    def config_key(cls) -> str:
        return "aggregation_type"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"value_int__{operation}_rolling_3"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        return {
            "in_features": "value_int",
            "frame_type": "rolling",
            "frame_size": 3,
            "partition_by": ["region"],
            "order_by": "timestamp",
        }

    @classmethod
    def pattern_match_options(cls) -> Options:
        return Options(context={"partition_by": ["region"], "order_by": "timestamp"})
