"""Tests for SqliteStringOps compute implementation."""

from __future__ import annotations

from typing import Any

import pytest
from mloda.user import Options
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

from mloda.community.feature_groups.data_operations.string.base import StringFeatureGroup
from mloda.community.feature_groups.data_operations.string.sqlite_string import (
    SqliteStringOps,
)
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin
from mloda.testing.feature_groups.data_operations.mixins.sqlite import SqliteTestMixin
from mloda.testing.feature_groups.data_operations.string.string import (
    StringTestBase,
)


class TestSqliteStringOps(CapabilityHookTestMixin, ReservedColumnsTestMixin, SqliteTestMixin, StringTestBase):
    """All tests inherited from the base class.

    SQLite supports only 'trim' and 'length' natively in a way that matches
    the PyArrow reference. 'upper'/'lower' are ASCII-only in SQLite and
    'reverse' has no native SQLite function, so all three are refused at
    match time and resolved by another framework.
    """

    @classmethod
    def supported_ops(cls) -> set[str]:
        return {"trim", "length"}

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteStringOps

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        return (("name__trim", Options()), ("name__length", Options()))

    @classmethod
    def capability_unsupported(cls) -> tuple[tuple[str, Options], ...]:
        return (
            ("name__upper", Options()),
            ("name__lower", Options()),
            ("name__reverse", Options()),
            ("my_upper", Options(context={"string_op": "upper", "in_features": "name"})),
        )

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        return "name__trim"

    @classmethod
    def reserved_columns_partition_by(cls) -> list[str] | None:
        return None


class TestSqliteUnsupportedOps:
    """SQLite refuses upper/lower/reverse at match time; the resolver falls
    back to another framework rather than silently producing ASCII-only output."""

    @pytest.mark.parametrize("operation", ["upper", "lower", "reverse"])
    def test_unsupported_op_does_not_match(self, operation: str) -> None:
        """The group does not support SqliteFramework for the operation."""
        assert not StringFeatureGroup.supports_compute_framework(f"name__{operation}", Options(), SqliteFramework)

    def test_supported_ops_still_match(self) -> None:
        """The group supports SqliteFramework for trim and length."""
        for op in ("trim", "length"):
            assert StringFeatureGroup.supports_compute_framework(f"name__{op}", Options(), SqliteFramework), (
                f"Expected name__{op} to be supported on SqliteFramework"
            )
