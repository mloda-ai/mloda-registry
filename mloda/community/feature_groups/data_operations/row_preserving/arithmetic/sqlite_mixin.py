"""Shared SQLite classmethods for the point- and scalar-arithmetic families.

``SqliteArithmeticMixin`` is a plain class supplying the framework's
``_non_numeric_descriptor``, which the point and scalar Implementations pass to
the group's ``apply_arithmetic``. One mixin per module keeps optional-dependency
isolation: this module imports only the SQLite backend.
"""

from __future__ import annotations

from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.sqlite_numeric_source import (
    sqlite_non_numeric_descriptor,
)


class SqliteArithmeticMixin:
    @classmethod
    def _non_numeric_descriptor(cls, data: SqliteRelation, source_col: str) -> object | None:
        """Report non-numeric source columns via ``PRAGMA table_info`` declared affinity.

        Caveat: ``SqliteRelation.from_arrow`` maps arrow booleans to SQLite
        ``INTEGER`` affinity (see ``mloda_plugins`` ``_arrow_type_to_sqlite``),
        so a boolean source column is indistinguishable from ``int64`` at the
        relation level. The shared tests ``test_boolean_source_column_rejected[_col_a/b]``
        are therefore skipped for SQLite (``detects_non_numeric_source`` override); the 0/1
        result is instead pinned by ``test_boolean_source_column_computed_when_undetected``
        and documented in ``docs/guides/data-operation-patterns/known-divergences.md``.
        """
        return sqlite_non_numeric_descriptor(data, source_col)
