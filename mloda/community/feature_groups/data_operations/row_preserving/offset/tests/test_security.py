"""Security tests for offset operations."""

from __future__ import annotations

from typing import Any

from mloda.testing.feature_groups.data_operations.security import SecurityTestBase

from mloda.community.feature_groups.data_operations.row_preserving.offset.base import (
    OffsetFeatureGroup,
)


class TestOffsetSecurity(SecurityTestBase):
    @classmethod
    def feature_group_class(cls) -> Any:
        return OffsetFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return {"first_value", "last_value", "lag_1", "lead_1"}

    @classmethod
    def config_key(cls) -> str:
        return "offset_type"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"value_int__{operation}_offset"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        return {"in_features": "value_int", "partition_by": ["region"], "order_by": "timestamp"}

    @classmethod
    def options_reject_invalid_types(cls) -> bool:
        return False
