"""Security tests for string operations."""

from __future__ import annotations

from typing import Any

from mloda.testing.feature_groups.data_operations.security import SecurityTestBase

from mloda.community.feature_groups.data_operations.string.base import (
    STRING_OPS,
    StringFeatureGroup,
)


class TestStringSecurity(SecurityTestBase):
    @classmethod
    def feature_group_class(cls) -> Any:
        return StringFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return set(STRING_OPS)

    @classmethod
    def config_key(cls) -> str:
        return "string_op"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"name__{operation}"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        return {"in_features": "name"}
