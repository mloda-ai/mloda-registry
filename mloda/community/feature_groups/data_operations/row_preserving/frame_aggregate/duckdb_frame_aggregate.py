"""DuckDB implementation for frame aggregate feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sql.sql_window import (
    CurrentRow,
    OrderBy,
    Preceding,
    Unbounded,
    WindowFrame,
)

from mloda.community.feature_groups.data_operations.duckdb_helpers import (
    nan_policy_agg_sql,
    nan_policy_window,
    preserve_row_order,
)
from mloda.community.feature_groups.data_operations.errors import (
    unsupported_agg_type_error,
    unsupported_frame_type_error,
)
from mloda.community.feature_groups.data_operations.mask_utils import build_sql_case_when
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


class DuckdbFrameAggregate(Implementation[FrameAggregateFeatureGroup]):
    feature_group = FrameAggregateFeatureGroup
    compute_framework = DuckDBFramework

    @classmethod
    def calculate(
        cls,
        group: type[FrameAggregateFeatureGroup],
        data: Any,
        features: FeatureSet,
        framework: ComputeFramework | None,
    ) -> Any:
        return group.aggregate(data, features, compute_frame=cls._compute_frame)

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
            source_sql = build_sql_case_when(DuckDBFramework.mask_engine(), data, mask_spec, quoted_source)
        quoted_order = quote_ident(order_by)

        if frame_type == "time":
            # Correlated subquery aliases the source table as "s"; build the wrapped
            # inner source SQL here, where source_col (needed for the type lookup) is
            # in scope, so the callee only ever sees the final SQL string.
            inner_source = f"s.{quoted_source}"
            if mask_spec is not None:
                inner_source = build_sql_case_when(DuckDBFramework.mask_engine(), data, mask_spec, inner_source)
            agg_sql = nan_policy_agg_sql(data, source_col, inner_source, agg_type, agg_func)
            return cls._compute_time_frame(
                data=data,
                feature_name=feature_name,
                agg_sql=agg_sql,
                partition_by=partition_by,
                quoted_order=quoted_order,
                frame_size=frame_size,
                frame_unit=frame_unit,
            )

        # NULLS LAST is equivalent to the old CASE WHEN order IS NULL THEN 1 ELSE 0 END
        # tiebreaker that sorted nulls after non-nulls within an ascending order.
        order_spec: list[OrderBy] = [OrderBy(order_by, nulls="last")]

        if frame_type in ("cumulative", "expanding"):
            frame = WindowFrame("rows", Unbounded(), CurrentRow())
        elif frame_type == "rolling":
            window_size = int(frame_size) if frame_size is not None else 1
            frame = WindowFrame("rows", Preceding(window_size - 1), CurrentRow())
        else:
            raise unsupported_frame_type_error(
                frame_type,
                FrameAggregateFeatureGroup.SUPPORTED_FRAME_TYPES,
                framework="DuckDB",
            )

        # PyArrow parity: the reference preserves input row order.
        return preserve_row_order(
            data,
            feature_name,
            lambda rel, rn: nan_policy_window(
                rel,
                source_col,
                source_sql,
                agg_type,
                agg_func,
                feature_name,
                partition_by=partition_by,
                order_by=order_spec,
                frame=frame,
            ),
        )

    @classmethod
    def _compute_time_frame(
        cls,
        data: DuckdbRelation,
        feature_name: str,
        agg_sql: str,
        partition_by: list[str],
        quoted_order: str,
        frame_size: int | None,
        frame_unit: str | None,
    ) -> DuckdbRelation:
        """Compute a time-based window aggregate via a correlated subquery.

        DuckDB ``RANGE BETWEEN INTERVAL '{N}' {unit} PRECEDING AND CURRENT ROW``
        is a clean one-liner but uses peer-set semantics: it includes every row
        whose ``order_by`` equals the current row's ``order_by``, even peers
        that come later in physical position. The reference uses
        ``rows[:pos+1]`` after a stable sort, excluding later peers. To match
        the reference, the input-order tag from ``preserve_row_order`` is the
        tiebreaker in a correlated subquery.

        When the outer row's ``order_by`` is NULL, the reference returns just
        the row's own source value (see reference.py:115-116). The NULL branch
        below matches only the self-row for that case.
        """
        size = int(frame_size) if frame_size is not None else 1
        unit = str(frame_unit or "day").upper()

        if partition_by:
            partition_eq = " AND ".join(
                f"(s.{quote_ident(col)} = t.{quote_ident(col)} "
                f"OR (s.{quote_ident(col)} IS NULL AND t.{quote_ident(col)} IS NULL))"
                for col in partition_by
            )
        else:
            partition_eq = "1=1"

        quoted_feature = quote_ident(feature_name)

        def compute(tagged: DuckdbRelation, rn: str) -> DuckdbRelation:
            qrn = quote_ident(rn)
            keep = ", ".join(quote_ident(c) for c in tagged.columns)

            # Safety: identifiers via quote_ident(); agg_sql built from a whitelisted
            # SQL function name; size/unit are sanitized integer/whitelisted-string values.
            sql = " ".join(  # nosec
                [
                    f"SELECT {keep},",
                    f"(SELECT {agg_sql}",
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
                ]
            )
            return tagged.query("tagged", sql)

        return preserve_row_order(data, feature_name, compute)
