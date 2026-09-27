"""Tests for the DuckDB float-only NaN-to-null wrap helper."""

from __future__ import annotations

import math
from typing import Any, Callable

import pytest

duckdb = pytest.importorskip("duckdb")

import pyarrow as pa
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

from mloda.community.feature_groups.data_operations.duckdb_helpers import nan_policy_agg_sql, nan_to_null_sql


class TestNanToNullSql:
    def test_float_type_wraps_with_nullif(self) -> None:
        """FLOAT/DOUBLE columns get NULLIF(expr, 'NaN'), so NaN counts as null in aggregate functions."""
        assert nan_to_null_sql("val", "DOUBLE") == "NULLIF(val, 'NaN')"
        assert nan_to_null_sql("val", "FLOAT") == "NULLIF(val, 'NaN')"

    def test_non_float_type_passes_through_unchanged(self) -> None:
        """A non-float column (e.g. TIMESTAMP, BIGINT) is returned unchanged."""
        assert nan_to_null_sql("ts", "TIMESTAMP") == "ts"
        assert nan_to_null_sql("n", "BIGINT") == "n"


# ---------------------------------------------------------------------------
# nan_policy_agg_sql: executed against a real DuckDB connection, both as a
# GROUP BY aggregate and as a cumulative window (last row == full window).
# ---------------------------------------------------------------------------

_WINDOW_OVER = ' OVER (ORDER BY "o" ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)'


def _relation(connection: Any, values: list[Any], dtype: pa.DataType = pa.float64()) -> DuckdbRelation:
    table = pa.table({"v": pa.array(values, type=dtype), "o": list(range(len(values)))})
    return DuckdbRelation.from_arrow(connection, table)


def _run_group(rel: DuckdbRelation, agg_sql: str) -> Any:
    result = rel.aggregate(f"{agg_sql} AS result")
    return result.to_arrow_table().column("result")[0].as_py()


def _run_window(rel: DuckdbRelation, agg_sql: str) -> Any:
    """Last row of the cumulative window equals the full-relation aggregate.

    ``agg_sql`` may be a struct-valued mode call ending in ``.v``; OVER must
    apply to the aggregate call itself, before the struct field access.
    """
    if agg_sql.endswith(".v"):
        base = agg_sql[: -len(".v")]
        window_sql = f"{base}{_WINDOW_OVER}.v"
    else:
        window_sql = f"{agg_sql}{_WINDOW_OVER}"
    result = rel.project(f"{window_sql} AS result")
    return result.to_arrow_table().column("result")[-1].as_py()


_RUNNERS: list[Callable[[DuckdbRelation, str], Any]] = [_run_group, _run_window]
_RUNNER_IDS = ["group_by", "window"]


class TestNanPolicyAggSql:
    """Tests for ``nan_policy_agg_sql``: min/max skip NaN, an all-NaN result stays NaN.

    Mode counts NaN as one value (via struct comparison), ties go to first
    occurrence, and nulls are skipped. See
    docs/guides/data-operation-patterns/03-reference-implementation.md.
    """

    def setup_method(self) -> None:
        self.conn = duckdb.connect()

    def teardown_method(self) -> None:
        self.conn.close()

    @pytest.mark.parametrize("run", _RUNNERS, ids=_RUNNER_IDS)
    def test_max_skips_nan(self, run: Callable[[DuckdbRelation, str], Any]) -> None:
        rel = _relation(self.conn, [1.0, float("nan"), 2.0])
        agg_sql = nan_policy_agg_sql(rel, "v", '"v"', "max", "MAX")
        assert run(rel, agg_sql) == 2.0

    @pytest.mark.parametrize("run", _RUNNERS, ids=_RUNNER_IDS)
    def test_max_all_nan_stays_nan(self, run: Callable[[DuckdbRelation, str], Any]) -> None:
        rel = _relation(self.conn, [float("nan"), float("nan")])
        agg_sql = nan_policy_agg_sql(rel, "v", '"v"', "max", "MAX")
        result = run(rel, agg_sql)
        assert result is not None and math.isnan(result)

    @pytest.mark.parametrize("run", _RUNNERS, ids=_RUNNER_IDS)
    def test_max_null_and_nan_stays_nan(self, run: Callable[[DuckdbRelation, str], Any]) -> None:
        rel = _relation(self.conn, [None, float("nan")])
        agg_sql = nan_policy_agg_sql(rel, "v", '"v"', "max", "MAX")
        result = run(rel, agg_sql)
        assert result is not None and math.isnan(result)

    @pytest.mark.parametrize("run", _RUNNERS, ids=_RUNNER_IDS)
    def test_max_all_null_stays_null(self, run: Callable[[DuckdbRelation, str], Any]) -> None:
        rel = _relation(self.conn, [None, None])
        agg_sql = nan_policy_agg_sql(rel, "v", '"v"', "max", "MAX")
        assert run(rel, agg_sql) is None

    def test_bigint_max_stays_unwrapped(self) -> None:
        """A non-float (BIGINT) column is not wrapped, same as today."""
        rel = _relation(self.conn, [1, 2, 3], dtype=pa.int64())
        agg_sql = nan_policy_agg_sql(rel, "v", '"v"', "max", "MAX")
        assert agg_sql == 'MAX("v")'

    @pytest.mark.parametrize("run", _RUNNERS, ids=_RUNNER_IDS)
    def test_min_all_nan_stays_nan_via_plain_min(self, run: Callable[[DuckdbRelation, str], Any]) -> None:
        """MIN needs no wrapping guard: DuckDB's plain MIN already skips NaN and stays NaN when all-NaN."""
        rel = _relation(self.conn, [float("nan"), float("nan")])
        agg_sql = nan_policy_agg_sql(rel, "v", '"v"', "min", "MIN")
        assert agg_sql == 'MIN("v")'
        result = run(rel, agg_sql)
        assert result is not None and math.isnan(result)

    @pytest.mark.parametrize("run", _RUNNERS, ids=_RUNNER_IDS)
    def test_median_skips_nan(self, run: Callable[[DuckdbRelation, str], Any]) -> None:
        rel = _relation(self.conn, [1.0, float("nan"), 3.0])
        agg_sql = nan_policy_agg_sql(rel, "v", '"v"', "median", "MEDIAN")
        assert run(rel, agg_sql) == pytest.approx(2.0)

    @pytest.mark.parametrize("run", _RUNNERS, ids=_RUNNER_IDS)
    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            pytest.param([1.0, float("nan"), float("nan")], float("nan"), id="one_real_two_nan"),
            pytest.param([float("nan"), 1.0, float("nan")], float("nan"), id="nan_real_nan"),
            pytest.param([1.0, float("nan"), float("nan"), 1.0], 1.0, id="tie_first_occurrence"),
            pytest.param([None, None, 1.0], 1.0, id="nulls_skipped"),
        ],
    )
    def test_mode_nan_policy(
        self, run: Callable[[DuckdbRelation, str], Any], values: list[Any], expected: float
    ) -> None:
        rel = _relation(self.conn, values)
        agg_sql = nan_policy_agg_sql(rel, "v", '"v"', "mode", "MODE") + ".v"
        result = run(rel, agg_sql)
        if math.isnan(expected):
            assert result is not None and math.isnan(result)
        else:
            assert result == expected
