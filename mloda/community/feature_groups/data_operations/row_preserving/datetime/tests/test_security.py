"""Security tests for datetime extraction operations."""

from __future__ import annotations

from typing import Any

from mloda.testing.feature_groups.data_operations.security import SecurityTestBase

from mloda.community.feature_groups.data_operations.row_preserving.datetime.base import (
    DATETIME_OPS,
    DateTimeFeatureGroup,
)


class TestDateTimeSecurity(SecurityTestBase):
    @classmethod
    def feature_group_class(cls) -> Any:
        return DateTimeFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return set(DATETIME_OPS)

    @classmethod
    def config_key(cls) -> str:
        return "datetime_op"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"timestamp__{operation}"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        return {"in_features": "timestamp"}
