"""Shared SQLite classmethods for the point- and scalar-arithmetic families.

``SqliteArithmeticMixin`` is a plain runtime class typed against
``ArithmeticFeatureGroupBase`` for mypy only, so it stays out of FeatureGroup
plugin discovery. It supplies ``compute_framework_rule``,
``_input_columns_and_framework``, and the ``_non_numeric_descriptor`` hook
consumed by the base's ``_assert_source_column_is_numeric`` template; the
structural guards live in ``tests/test_numeric_source.py``. The concrete
SQLite backends import ``SQLITE_ARITHMETIC_OPS`` from here (an alias of the
shared ``SQL_ARITHMETIC_OPS``). Keeping one mixin per module preserves
optional-dependency isolation: this module imports only the SQLite backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.arithmetic.base import SQL_ARITHMETIC_OPS
from mloda.community.feature_groups.data_operations.arithmetic.sqlite_numeric_source import (
    sqlite_non_numeric_descriptor,
)

if TYPE_CHECKING:
    from mloda.community.feature_groups.data_operations.arithmetic.base import ArithmeticFeatureGroupBase

    _ArithmeticMixinBase = ArithmeticFeatureGroupBase
else:
    _ArithmeticMixinBase = object

SQLITE_ARITHMETIC_OPS: dict[str, str] = SQL_ARITHMETIC_OPS


class SqliteArithmeticMixin(_ArithmeticMixinBase):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SqliteFramework}

    @classmethod
    def _input_columns_and_framework(cls, data: SqliteRelation) -> tuple[list[str], str]:
        return list(data.columns), "SQLite"

    @classmethod
    def _non_numeric_descriptor(cls, data: SqliteRelation, source_col: str) -> object | None:
        """Report non-numeric source columns via ``PRAGMA table_info`` declared affinity.

        Caveat: ``SqliteRelation.from_arrow`` maps arrow booleans to SQLite
        ``INTEGER`` affinity (see ``mloda_plugins`` ``_arrow_type_to_sqlite``),
        so a boolean source column is indistinguishable from ``int64`` at the
        relation level. The shared tests ``test_boolean_source_column_rejected[_col_a/b]``
        are correspondingly skipped for SQLite via the
        ``detects_non_numeric_source`` test-class override.
        """
        return sqlite_non_numeric_descriptor(data, source_col)
