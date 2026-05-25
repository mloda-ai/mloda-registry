"""SQLite implementation for frame aggregate feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.aggregate_helpers import aggregate
from mloda.community.feature_groups.data_operations.errors import (
    unsupported_agg_type_error,
    unsupported_frame_type_error,
)
from mloda.community.feature_groups.data_operations.mask_utils import apply_pyarrow_mask, build_sql_case_when
from mloda.community.feature_groups.data_operations.reserved_columns import assert_no_reserved_columns
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
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SqliteFramework}

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
        assert_no_reserved_columns(data.columns, framework="SQLite", operation="frame aggregate")

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
        partition_clause = ", ".join(quote_ident(col) for col in partition_by)
        quoted_feature = quote_ident(feature_name)
        qrn = quote_ident("__mloda_rn__")

        null_sort = f"CASE WHEN {quoted_order} IS NULL THEN 1 ELSE 0 END"
        order_clause = f"{null_sort}, {quoted_order}"

        if frame_type == "time":
            unit = str(frame_unit or "day")
            if unit in ("month", "year"):
                # Month/year use calendar arithmetic (relativedelta) for end-of-month
                # parity with the reference. SQLite's native ``datetime(ts, '-N months')``
                # uses fixed-day-of-month rollover (Mar 31 -1mo = Mar 3) which diverges
                # from ``dateutil.relativedelta`` (Mar 31 -1mo = Feb 28). Compute in Python.
                return cls._compute_time_frame_python(
                    data=data,
                    feature_name=feature_name,
                    source_col=source_col,
                    partition_by=partition_by,
                    order_by=order_by,
                    agg_type=agg_type,
                    frame_size=frame_size,
                    frame_unit=unit,
                    mask_spec=mask_spec,
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

        if frame_type in ("cumulative", "expanding"):
            frame_clause = "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
        elif frame_type == "rolling":
            window_size = int(frame_size) if frame_size is not None else 1
            frame_clause = f"ROWS BETWEEN {window_size - 1} PRECEDING AND CURRENT ROW"
        else:
            raise unsupported_frame_type_error(
                frame_type,
                cls.SUPPORTED_FRAME_TYPES,
                framework="SQLite",
            )

        # Safety: all identifiers use quote_ident(), agg_func from whitelist
        sql = " ".join(  # nosec
            [
                "SELECT",
                f"{agg_func}({source_sql}) OVER",
                f"(PARTITION BY {partition_clause} ORDER BY {order_clause} {frame_clause})",
                f"AS {quoted_feature},",
                f"ROW_NUMBER() OVER (ORDER BY rowid) AS {qrn}",
                "FROM",
                f"{quote_ident(data.table_name)}",
                "ORDER BY",
                qrn,
            ]
        )
        cursor = data.connection.execute(sql)
        rows = cursor.fetchall()

        result_values = [row[0] for row in rows]
        return data.append_column(feature_name, result_values)

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

        Only second/minute/hour/day/week units reach this path; month/year are
        routed to ``_compute_time_frame_python`` for calendar-anchored arithmetic.
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
        qrn = quote_ident("__mloda_rn__")
        quoted_table = quote_ident(data.table_name)

        # Safety: identifiers via quote_ident(); agg_func from whitelist; n_days is
        # computed in Python from sanitized integer/unit values, embedded as a
        # numeric literal.
        #
        # When the outer row's order_by is NULL, ``julianday(NULL)`` is NULL and the
        # BETWEEN short-circuits to NULL, which would leave the row with a NULL aggregate.
        # The PyArrow reference returns the source value of just the current row in that
        # case (see reference.py:115-116). The OR branch below matches the self-row only
        # when ``t.{order_by}`` is NULL, restoring reference parity.
        sql = " ".join(  # nosec
            [
                "SELECT",
                "(SELECT",
                f"{agg_func}({inner_source_sql})",
                f"FROM {quoted_table} s",
                f"WHERE {partition_eq}",
                "AND (",
                f"(t.{quoted_order} IS NOT NULL AND s.{quoted_order} IS NOT NULL",
                f"AND julianday(s.{quoted_order}) BETWEEN julianday(t.{quoted_order}) - {n_days}",
                f"AND julianday(t.{quoted_order}))",
                f"OR (t.{quoted_order} IS NULL AND s.rowid = t.rowid)",
                ")",
                f") AS {quoted_feature},",
                f"ROW_NUMBER() OVER (ORDER BY t.rowid) AS {qrn}",
                f"FROM {quoted_table} t",
                "ORDER BY",
                qrn,
            ]
        )
        cursor = data.connection.execute(sql)
        rows = cursor.fetchall()

        result_values = [row[0] for row in rows]
        return data.append_column(feature_name, result_values)

    @classmethod
    def _compute_time_frame_python(
        cls,
        data: SqliteRelation,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
        agg_type: str,
        frame_size: int | None,
        frame_unit: str,
        mask_spec: list[tuple[str, str, Any]] | None,
    ) -> SqliteRelation:
        """Calendar-anchored time window (month/year) computed in Python.

        SQLite's ``datetime(ts, '-N months')`` uses native rollover (Mar 31 -1mo
        = Mar 3) which diverges from ``dateutil.relativedelta`` (Mar 31 -1mo =
        Feb 28) used by the PyArrow reference. Read the table into Arrow, apply
        the mask, group by partition, sort each group with nulls last, then for
        each row compute the relativedelta-based window and aggregate in pure
        Python (mirrors ``reference.py:_time_window``).
        """
        from datetime import date, datetime

        from dateutil.relativedelta import relativedelta

        size = int(frame_size) if frame_size is not None else 1
        delta = relativedelta(months=size) if frame_unit == "month" else relativedelta(years=size)

        table = data.to_arrow_table()
        if mask_spec is not None:
            table = apply_pyarrow_mask(table, source_col, mask_spec)

        num_rows = table.num_rows
        partition_lists = {col: table.column(col).to_pylist() for col in partition_by}

        # SQLite stores datetimes via its text adapter, so ``to_arrow_table()``
        # returns them as ISO-8601 strings. Parse back to ``datetime`` so that
        # ``relativedelta`` arithmetic works.
        def _parse_ts(v: Any) -> Any:
            if v is None or isinstance(v, (datetime, date)):
                return v
            if isinstance(v, str):
                try:
                    return datetime.fromisoformat(v)
                except ValueError:
                    return v
            return v

        order_vals = [_parse_ts(v) for v in table.column(order_by).to_pylist()]
        source_vals = table.column(source_col).to_pylist()

        groups: dict[tuple[Any, ...], list[tuple[int, Any, Any]]] = {}
        for i in range(num_rows):
            key = tuple(partition_lists[col][i] for col in partition_by)
            groups.setdefault(key, []).append((i, order_vals[i], source_vals[i]))

        for key in groups:
            groups[key].sort(key=lambda t: (t[1] is None, t[1] if t[1] is not None else 0))

        result_values: list[Any] = [None] * num_rows
        for rows in groups.values():
            for pos, (orig_idx, order_val, _) in enumerate(rows):
                if order_val is None:
                    window = [rows[pos][2]]
                else:
                    cutoff = order_val - delta
                    window = [r[2] for r in rows[: pos + 1] if r[1] is not None and r[1] >= cutoff]
                result_values[orig_idx] = aggregate(window, agg_type)

        return data.append_column(feature_name, result_values)
