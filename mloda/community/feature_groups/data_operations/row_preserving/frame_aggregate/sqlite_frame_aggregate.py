"""SQLite implementation for frame aggregate feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import pick_helper_column_name, quote_ident
from mloda_plugins.compute_framework.base_implementations.sql.sql_window import (
    CurrentRow,
    OrderBy,
    Preceding,
    Unbounded,
    WindowFrame,
)
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.errors import (
    unsupported_agg_type_error,
    unsupported_frame_type_error,
)
from mloda.community.feature_groups.data_operations.mask_utils import build_sql_case_when
from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
    FrameAggregateFeatureGroup,
)

_SQLITE_AGG_FUNCS: dict[str, str] = {
    "sum": "SUM",
    "avg": "AVG",
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
}


class SqliteFrameAggregate(FrameAggregateFeatureGroup):
    # SQLite has no native calendar-anchored INTERVAL arithmetic: ``datetime(ts, '-N months')``
    # uses fixed-day-of-month rollover (Mar 31 -1mo = Mar 3) which diverges from the
    # ``dateutil.relativedelta`` semantics (Mar 31 -1mo = Feb 28) used by the reference
    # implementation. Rather than fall back to Python-side calendar arithmetic (which
    # would defeat the point of running inside the SQLite engine), month/year units are
    # rejected at match time. See known-divergences.md.
    SUPPORTED_TIME_UNITS: set[str] = {"second", "minute", "hour", "day", "week"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SqliteFramework}

    @classmethod
    def supported_subtypes(cls, secondary: str | None = None) -> frozenset[str] | None:
        """SQLite supports sum/avg/count/min/max for every frame type."""
        return frozenset(_SQLITE_AGG_FUNCS)

    @classmethod
    def _compute_frame(
        cls,
        data: SqliteRelation,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
        agg_type: str,
        frame_type: str,
        frame_size: int | None = None,
        frame_unit: str | None = None,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> SqliteRelation:
        agg_func = _SQLITE_AGG_FUNCS.get(agg_type)
        if agg_func is None:
            raise unsupported_agg_type_error(
                agg_type,
                _SQLITE_AGG_FUNCS.keys(),
                framework="SQLite",
                operation="frame aggregate",
            )

        quoted_source = quote_ident(source_col)
        source_sql = quoted_source
        if mask_spec is not None:
            source_sql = build_sql_case_when(mask_spec, quoted_source)
        quoted_order = quote_ident(order_by)

        if frame_type == "time":
            # Mask + source_col == order_by: the reference treats masked rows as having
            # null order_by (mask writes null into source_col, which is also order_by).
            # The SQLite correlated subquery uses the unmasked order_by for window bounds
            # even when ``CASE WHEN ... THEN source END`` wraps the aggregate expression,
            # so this combo cannot be expressed natively. Reject to match pandas.
            # See known-divergences.md.
            if mask_spec is not None and source_col == order_by:
                raise ValueError(
                    "SQLite frame aggregate (time frame): mask + source_col == order_by "
                    f"({source_col!r}) is unsupported. The reference semantic requires "
                    "treating masked rows as having null order_by, which the correlated "
                    "subquery cannot express natively. See known-divergences.md."
                )
            return cls._compute_time_frame(
                data=data,
                feature_name=feature_name,
                quoted_source=quoted_source,
                partition_by=partition_by,
                quoted_order=quoted_order,
                agg_func=agg_func,
                frame_size=frame_size,
                frame_unit=frame_unit,
                mask_spec=mask_spec,
            )

        frame: WindowFrame
        if frame_type in ("cumulative", "expanding"):
            frame = WindowFrame("rows", Unbounded(), CurrentRow())
        elif frame_type == "rolling":
            window_size = int(frame_size) if frame_size is not None else 1
            frame = WindowFrame("rows", Preceding(window_size - 1), CurrentRow())
        else:
            raise unsupported_frame_type_error(
                frame_type,
                cls.SUPPORTED_FRAME_TYPES,
                framework="SQLite",
            )

        # NullPolicy.NULLS_LAST: ``OrderBy(order_by, nulls="last")`` renders
        # ``ORDER BY ... NULLS LAST``, equivalent to the old
        # ``CASE WHEN order IS NULL THEN 1 ELSE 0 END, order`` sort key.
        original_cols = list(data.columns)
        rn = pick_helper_column_name(taken=set(data.columns) | {feature_name})
        rel = data.with_row_number(rn, order_by=["rowid"])
        rel = rel.window(
            f"{agg_func}({source_sql})",
            feature_name,
            partition_by=partition_by,
            order_by=[OrderBy(order_by, nulls="last")],
            frame=frame,
        )
        rel = rel.order(rn)
        return rel.select(*original_cols, feature_name)

    @classmethod
    def _compute_time_frame(
        cls,
        data: SqliteRelation,
        feature_name: str,
        quoted_source: str,
        partition_by: list[str],
        quoted_order: str,
        agg_func: str,
        frame_size: int | None,
        frame_unit: str | None,
        mask_spec: list[tuple[str, str, Any]] | None,
    ) -> SqliteRelation:
        """Compute a time-based window aggregate via a correlated subquery.

        SQLite lacks RANGE BETWEEN INTERVAL window frames, so for each row ``t``
        the inner SELECT aggregates rows ``s`` in the same partition whose
        ``order_by`` value falls in ``[t.order_by - delta, t.order_by]``.

        Bounds use ``julianday()`` arithmetic (a floating-point day count)
        rather than ``datetime(ts, '-N units')``. The latter strips fractional
        seconds and would lose sub-second precision; the former preserves
        microseconds via the fractional component of the Julian day.

        Only second/minute/hour/day/week units reach this path. Month/year are
        rejected at ``match_feature_group_criteria`` time via
        ``SUPPORTED_TIME_UNITS`` because SQLite's native month/year arithmetic
        diverges from ``relativedelta``. See known-divergences.md.
        """
        size = int(frame_size) if frame_size is not None else 1
        unit = str(frame_unit or "day")
        # Convert window span to fractional days for julianday() arithmetic.
        unit_to_days: dict[str, float] = {
            "second": size / 86400.0,
            "minute": size / 1440.0,
            "hour": size / 24.0,
            "day": float(size),
            "week": float(size * 7),
        }
        n_days = unit_to_days[unit]

        # Build the inner aggregate expression with the ``s.`` alias prefix.
        inner_source = f"s.{quoted_source}"
        inner_source_sql = build_sql_case_when(mask_spec, inner_source) if mask_spec is not None else inner_source

        if partition_by:
            partition_eq = " AND ".join(
                f"(s.{quote_ident(col)} = t.{quote_ident(col)} "
                f"OR (s.{quote_ident(col)} IS NULL AND t.{quote_ident(col)} IS NULL))"
                for col in partition_by
            )
        else:
            partition_eq = "1=1"

        quoted_feature = quote_ident(feature_name)
        quoted_table = quote_ident(data.table_name)

        # Safety: identifiers via quote_ident(); agg_func from whitelist; n_days is
        # computed in Python from sanitized integer/unit values, embedded as a
        # numeric literal.
        #
        # Peer handling: the PyArrow reference uses ``rows[:pos+1]`` after a stable
        # sort by order_by, so peers at the same order_by but later physical position
        # are excluded from the current row's window. SQL ``BETWEEN`` would include
        # all same-ts peers (both earlier and later), so the upper bound is split into
        # ``s.ts < t.ts`` plus an explicit rowid tiebreaker for the equal case. ``rowid``
        # is the stable insertion order in SQLite, matching the reference's stable-sort
        # tie-break behaviour.
        #
        # When the outer row's order_by is NULL, ``julianday(NULL)`` is NULL and the
        # comparison short-circuits to NULL, which would leave the row with a NULL
        # aggregate. The PyArrow reference returns the source value of just the current
        # row in that case (see reference.py:115-116). The OR branch matches the
        # self-row only when ``t.{order_by}`` is NULL, restoring reference parity.
        sql = " ".join(  # nosec
            [
                "SELECT",
                "(SELECT",
                f"{agg_func}({inner_source_sql})",
                f"FROM {quoted_table} s",
                f"WHERE {partition_eq}",
                "AND (",
                f"(t.{quoted_order} IS NOT NULL AND s.{quoted_order} IS NOT NULL",
                f"AND julianday(s.{quoted_order}) >= julianday(t.{quoted_order}) - {n_days}",
                f"AND (julianday(s.{quoted_order}) < julianday(t.{quoted_order})",
                f"OR (julianday(s.{quoted_order}) = julianday(t.{quoted_order}) AND s.rowid <= t.rowid)))",
                f"OR (t.{quoted_order} IS NULL AND s.rowid = t.rowid)",
                ")",
                f") AS {quoted_feature}",
                f"FROM {quoted_table} t",
                "ORDER BY t.rowid",
            ]
        )
        cursor = data.connection.execute(sql)
        rows = cursor.fetchall()

        result_values = [row[0] for row in rows]
        return data.append_column(feature_name, result_values)
