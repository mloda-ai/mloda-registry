"""Shared SQLite classmethods for the point- and scalar-arithmetic families.

The SQLite point and scalar arithmetic backends carry byte-for-byte-identical
``compute_framework_rule`` / ``_input_columns_and_framework`` /
``_assert_source_column_is_numeric`` implementations, plus the same
``SQLITE_ARITHMETIC_OPS`` operator map. This mixin holds the one copy; the
concrete SQLite backends inherit it (first in their MRO), import
``SQLITE_ARITHMETIC_OPS`` from here, and supply only ``_compute_arithmetic``.
Keeping one mixin per module preserves optional-dependency isolation: this
module imports only the SQLite backend.
"""

from __future__ import annotations

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.arithmetic_base import ArithmeticFeatureGroupBase
from mloda.community.feature_groups.data_operations.sqlite_numeric_source import sqlite_non_numeric_descriptor

SQLITE_ARITHMETIC_OPS: dict[str, str] = {
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "divide": "/",
}


class SqliteArithmeticMixin(ArithmeticFeatureGroupBase):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SqliteFramework}

    @classmethod
    def _input_columns_and_framework(cls, data: SqliteRelation) -> tuple[list[str], str]:
        return list(data.columns), "SQLite"

    @classmethod
    def _assert_source_column_is_numeric(cls, data: SqliteRelation, source_col: str) -> None:
        """Reject non-numeric source columns via ``PRAGMA table_info`` declared affinity.

        Caveat: ``SqliteRelation.from_arrow`` maps arrow booleans to SQLite
        ``INTEGER`` affinity (see ``mloda_plugins`` ``_arrow_type_to_sqlite``),
        so a boolean source column is indistinguishable from ``int64`` at the
        relation level. The shared tests ``test_boolean_source_column_rejected[_col_a/b]``
        are correspondingly skipped for SQLite via the
        ``detects_non_numeric_source`` test-class override.
        """
        descriptor = sqlite_non_numeric_descriptor(data, source_col)
        if descriptor is not None:
            cls._raise_non_numeric_source(source_col, descriptor)
