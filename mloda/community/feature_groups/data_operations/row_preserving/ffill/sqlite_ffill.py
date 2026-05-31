"""SQLite implementation of ffill-by-time."""

from __future__ import annotations

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import pick_helper_column_name, quote_ident
from mloda_plugins.compute_framework.base_implementations.sql.sql_window import (
    CurrentRow,
    OrderBy,
    Unbounded,
    WindowFrame,
)
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.ffill.base import FfillFeatureGroup


class SqliteFfill(FfillFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SqliteFramework}

    @classmethod
    def _assert_source_column_present(cls, data: SqliteRelation, source_col: str) -> None:
        if source_col not in data.columns:
            raise ValueError(
                f"Source column {source_col!r} is not present in the SQLite relation; available: {data.columns}."
            )

    @classmethod
    def _compute_ffill(
        cls,
        data: SqliteRelation,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
    ) -> SqliteRelation:
        original_cols = list(data.columns)

        # SQLite does not support LAST_VALUE(... IGNORE NULLS), so ffill is built
        # from the classic two-window "fill group" idiom, both windows expressed
        # via the typed window helper:
        #   1. grp = COUNT(src) over (... ROWS UNBOUNDED PRECEDING .. CURRENT ROW)
        #      counts non-nulls seen so far; leading nulls get grp = 0.
        #   2. result = MAX(src) over (PARTITION BY [*partition_by, grp])
        #      picks the single non-null value of each fill group (the grp = 0
        #      leading-null group has no non-null -> MAX stays NULL).
        taken = set(data.columns) | {feature_name}
        rn = pick_helper_column_name(taken=taken)
        grp = pick_helper_column_name(taken=taken | {rn})

        quoted_source = quote_ident(source_col)

        # Tag rows with original order (rowid is stable insertion order in SQLite).
        rel = data.with_row_number(rn, order_by=["rowid"])

        # Step 1: fill-group id via a running COUNT over the ordered partition.
        rel = rel.window(
            f"COUNT({quoted_source})",
            grp,
            partition_by=partition_by,
            order_by=[OrderBy(order_by, nulls="last")],
            frame=WindowFrame("rows", Unbounded(), CurrentRow()),
        )

        # Step 2: carry the single non-null value across each fill group.
        rel = rel.window(
            f"MAX({quoted_source})",
            feature_name,
            partition_by=[*partition_by, grp],
        )

        # Restore original row order and drop both helper columns.
        rel = rel.order(rn)
        return rel.select(*original_cols, feature_name)
