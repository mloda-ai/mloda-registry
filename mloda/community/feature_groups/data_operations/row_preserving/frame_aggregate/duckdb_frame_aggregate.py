"""DuckDB implementation for frame aggregate feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.errors import (
    unsupported_agg_type_error,
    unsupported_frame_type_error,
)
from mloda.community.feature_groups.data_operations.mask_utils import build_sql_case_when
from mloda.community.feature_groups.data_operations.reserved_columns import assert_no_reserved_columns
from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
    FrameAggregateFeatureGroup,
)

_DUCKDB_AGG_FUNCS: dict[str, str] = {
    "sum": "SUM",
    "avg": "AVG",
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
    "std": "STDDEV_POP",
    "var": "VAR_POP",
    "median": "MEDIAN",
}

_RN_COL = "__mloda_rn__"


class DuckdbFrameAggregate(FrameAggregateFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {DuckDBFramework}

    @classmethod
    def _compute_frame(
        cls,
        data: DuckdbRelation,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
        agg_type: str,
        frame_type: str,
        frame_size: int | None = None,
        frame_unit: str | None = None,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> DuckdbRelation:
        assert_no_reserved_columns(data.columns, framework="DuckDB", operation="frame aggregate")

        agg_func = _DUCKDB_AGG_FUNCS.get(agg_type)
        if agg_func is None:
            raise unsupported_agg_type_error(
                agg_type,
                _DUCKDB_AGG_FUNCS.keys(),
                framework="DuckDB",
                operation="frame aggregate",
            )

        quoted_source = quote_ident(source_col)
        source_sql = quoted_source
        if mask_spec is not None:
            source_sql = build_sql_case_when(mask_spec, quoted_source)
        quoted_feature = quote_ident(feature_name)
        partition_clause = ", ".join(quote_ident(col) for col in partition_by)
        quoted_order = quote_ident(order_by)
        qrn = quote_ident(_RN_COL)

        if frame_type == "time":
            # Mask + source_col == order_by: the reference treats masked rows as having
            # null order_by (mask writes null into source_col, which is also order_by).
            # The DuckDB correlated subquery uses the unmasked order_by for window bounds
            # even when ``CASE WHEN ... THEN source END`` wraps the aggregate expression,
            # so this combo cannot be expressed natively. Reject to match pandas.
            # See known-divergences.md.
            if mask_spec is not None and source_col == order_by:
                raise ValueError(
                    "DuckDB frame aggregate (time frame): mask + source_col == order_by "
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

        null_sort = f"CASE WHEN {quoted_order} IS NULL THEN 1 ELSE 0 END"
        order_clause = f"{null_sort}, {quoted_order}"

        if frame_type in ("cumulative", "expanding"):
            frame_clause = "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
        elif frame_type == "rolling":
            window_size = int(frame_size) if frame_size is not None else 1
            frame_clause = f"ROWS BETWEEN {window_size - 1} PRECEDING AND CURRENT ROW"
        else:
            raise unsupported_frame_type_error(
                frame_type,
                cls.SUPPORTED_FRAME_TYPES,
                framework="DuckDB",
            )

        # PyArrow parity: the reference preserves input row order. DuckDB
        # ORDER BY in the window frame reorders rows; tag with ROW_NUMBER(),
        # compute, then .order(qrn) to restore original order.
        # Step 1: tag rows with original position
        rel = data.project(f"*, ROW_NUMBER() OVER () AS {qrn}")  # nosec

        # Step 2: compute window function with frame
        raw_sql = (  # nosec
            f"*, {agg_func}({source_sql}) OVER "
            f"(PARTITION BY {partition_clause} ORDER BY {order_clause} {frame_clause}) "
            f"AS {quoted_feature}"
        )
        rel = rel.project(raw_sql)

        # Step 3: restore original order, drop helper
        rel = rel.order(qrn)
        keep = ", ".join(quote_ident(c) for c in rel.columns if c != _RN_COL)
        rel = rel.project(keep)

        return rel

    @classmethod
    def _compute_time_frame(
        cls,
        data: DuckdbRelation,
        feature_name: str,
        quoted_source: str,
        partition_by: list[str],
        quoted_order: str,
        agg_func: str,
        frame_size: int | None,
        frame_unit: str | None,
        mask_spec: list[tuple[str, str, Any]] | None,
    ) -> DuckdbRelation:
        """Compute a time-based window aggregate via a correlated subquery.

        DuckDB ``RANGE BETWEEN INTERVAL '{N}' {unit} PRECEDING AND CURRENT ROW``
        is a clean one-liner but uses peer-set semantics: it includes every row
        whose ``order_by`` equals the current row's ``order_by``, even peers
        that come later in physical position. The PyArrow reference uses
        ``rows[:pos+1]`` after a stable sort, excluding later peers. To match
        the reference, this method tags rows with ``ROW_NUMBER() OVER ()`` and
        uses that tag as a tiebreaker in a correlated subquery.

        When the outer row's ``order_by`` is NULL, the reference returns just
        the row's own source value (see reference.py:115-116). The NULL branch
        below matches only the self-row for that case.
        """
        size = int(frame_size) if frame_size is not None else 1
        unit = str(frame_unit or "day").upper()
        qrn = quote_ident(_RN_COL)

        # Step 1: tag rows with original position; this is also the tiebreaker.
        tagged = data.project(f"*, ROW_NUMBER() OVER () AS {qrn}")  # nosec

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
        keep = ", ".join(quote_ident(c) for c in data.columns)

        # Safety: identifiers via quote_ident(); agg_func from whitelist; size/unit
        # are sanitized integer/whitelisted-string values.
        sql = " ".join(  # nosec
            [
                f"SELECT {keep},",
                f"(SELECT {agg_func}({inner_source_sql})",
                "FROM tagged s",
                f"WHERE {partition_eq}",
                "AND (",
                f"(t.{quoted_order} IS NOT NULL AND s.{quoted_order} IS NOT NULL",
                f"AND s.{quoted_order} >= t.{quoted_order} - INTERVAL '{size}' {unit}",
                f"AND (s.{quoted_order} < t.{quoted_order}",
                f"OR (s.{quoted_order} = t.{quoted_order} AND s.{qrn} <= t.{qrn})))",
                f"OR (t.{quoted_order} IS NULL AND s.{qrn} = t.{qrn})",
                ")",
                f") AS {quoted_feature}",
                "FROM tagged t",
                f"ORDER BY t.{qrn}",
            ]
        )
        return tagged.query("tagged", sql)
