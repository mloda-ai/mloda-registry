"""Point-arithmetic-specific security tests: compute-level rejection, backend
allowlist completeness, per-row divide-by-zero handling, missing/extra in_features,
reserved-column guard, and source-column dtype validation.

Generic match-validation tests live in ``test_base.py`` via
``MatchValidationTestBase``.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
    ARITHMETIC_OPERATIONS,
)
from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.pyarrow_point_arithmetic import (
    PyArrowPointArithmetic,
)


def _make_fs(
    name: str,
    *,
    op: str | None = None,
    in_features: list[str] | None = None,
) -> FeatureSet:
    """Build a FeatureSet for a point arithmetic feature.

    If ``op`` is given, set arithmetic_op context (option-based).
    If ``in_features`` is given, set in_features context.
    """
    context: dict[str, Any] = {}
    if op is not None:
        context["arithmetic_op"] = op
    if in_features is not None:
        context["in_features"] = in_features
    feature = Feature(name, options=Options(context=context))
    fs = FeatureSet()
    fs.add(feature)
    return fs


class TestPointArithmeticComputeRejection:
    """Verify that invalid ops reaching compute raise ValueError."""

    def test_pandas_rejects_unknown_op_at_compute(self) -> None:
        pytest.importorskip("pandas")
        from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.pandas_point_arithmetic import (
            PandasPointArithmetic,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        df = arrow_table.to_pandas()
        with pytest.raises(ValueError, match="[Uu]nsupported"):
            PandasPointArithmetic._compute_arithmetic(df, "result_col", "value_int", "amount", "evil_op")

    def test_pyarrow_rejects_unknown_op_at_compute(self) -> None:
        arrow_table = PyArrowDataOpsTestDataCreator.create()
        with pytest.raises(ValueError, match="[Uu]nsupported"):
            PyArrowPointArithmetic._compute_arithmetic(arrow_table, "result_col", "value_int", "amount", "evil_op")


class TestAllowlistCompleteness:
    """Verify that every op in ARITHMETIC_OPERATIONS is covered by every backend."""

    def test_pandas_covers_all_ops(self) -> None:
        pytest.importorskip("pandas")
        from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.pandas_point_arithmetic import (
            PandasPointArithmetic,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        df = arrow_table.to_pandas()
        for op in ARITHMETIC_OPERATIONS:
            result = PandasPointArithmetic._compute_arithmetic(df, f"test_{op}", "value_int", "amount", op)
            assert f"test_{op}" in result.columns

    def test_pyarrow_covers_all_ops(self) -> None:
        arrow_table = PyArrowDataOpsTestDataCreator.create()
        for op in ARITHMETIC_OPERATIONS:
            result = PyArrowPointArithmetic._compute_arithmetic(arrow_table, f"test_{op}", "value_int", "amount", op)
            assert f"test_{op}" in result.column_names

    def test_polars_covers_all_ops(self) -> None:
        polars = pytest.importorskip("polars")
        from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.polars_lazy_point_arithmetic import (
            PolarsLazyPointArithmetic,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        lf = polars.from_arrow(arrow_table).lazy()
        for op in ARITHMETIC_OPERATIONS:
            result = PolarsLazyPointArithmetic._compute_arithmetic(lf, f"test_{op}", "value_int", "amount", op)
            collected = result.collect()
            assert f"test_{op}" in collected.columns

    def test_duckdb_covers_all_ops(self) -> None:
        pytest.importorskip("duckdb")
        from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.duckdb_point_arithmetic import (
            DUCKDB_ARITHMETIC_OPS,
        )

        for op in ARITHMETIC_OPERATIONS:
            assert op in DUCKDB_ARITHMETIC_OPS, f"DuckDB backend missing arithmetic op: {op}"

    def test_sqlite_covers_all_ops(self) -> None:
        from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.sqlite_point_arithmetic import (
            SQLITE_ARITHMETIC_OPS,
        )

        for op in ARITHMETIC_OPERATIONS:
            assert op in SQLITE_ARITHMETIC_OPS, f"SQLite backend missing arithmetic op: {op}"


def _assert_inf(value: Any, sign: int, *, row: int) -> None:
    """Assert ``value`` is an IEEE-754 inf with the given sign (+1 or -1)."""
    assert value is not None and isinstance(value, float) and math.isinf(value), (
        f"row {row}: expected inf, got {value!r}"
    )
    actual_sign = 1 if value > 0 else -1
    assert actual_sign == sign, f"row {row}: expected sign {sign}, got {value!r}"


def _assert_nan(value: Any, *, row: int) -> None:
    """Assert ``value`` is an IEEE-754 NaN."""
    assert value is not None and isinstance(value, float) and math.isnan(value), (
        f"row {row}: expected nan, got {value!r}"
    )


class TestDivideByZeroPerRow:
    """Per-backend divide-by-zero truth table.

    Reference truth table (verified empirically) using a 6-row pair:

    | row | a   | b   | PyArrow / Pandas / Polars / DuckDB | SQLite |
    |-----|-----|-----|------------------------------------|--------|
    | 0   | 10  | 2.0 | 5.0                                | 5.0    |
    | 1   | 10  | 0.0 | +inf                               | None   |
    | 2   | 0   | 0.0 | nan                                | None   |
    | 3   |None | 5.0 | None                               | None   |
    | 4   | 5   |None | None                               | None   |
    | 5   |-10  | 0.0 | -inf                               | None   |
    """

    @staticmethod
    def _truth_table_arrow() -> Any:
        """Build a 6-row arrow table with col_a (int), col_b (float)."""
        import pyarrow as pa

        col_a: list[int | None] = [10, 10, 0, None, 5, -10]
        col_b: list[float | None] = [2.0, 0.0, 0.0, 5.0, None, 0.0]
        return pa.table({"a": pa.array(col_a, type=pa.int64()), "b": pa.array(col_b, type=pa.float64())})

    @staticmethod
    def _make_divide_fs() -> FeatureSet:
        return _make_fs("a&b__divide_point")

    def _assert_inf_nan_table(self, values: list[Any]) -> None:
        """Common per-row assertions for PyArrow/Pandas/Polars/DuckDB."""
        assert values[0] == pytest.approx(5.0)
        _assert_inf(values[1], sign=1, row=1)
        _assert_nan(values[2], row=2)
        assert values[3] is None
        assert values[4] is None
        _assert_inf(values[5], sign=-1, row=5)

    def _assert_null_for_zero_table(self, values: list[Any]) -> None:
        """SQLite per-row assertions: NULL for any divide-by-zero or null operand."""
        assert values[0] == pytest.approx(5.0)
        assert values[1] is None
        assert values[2] is None
        assert values[3] is None
        assert values[4] is None
        assert values[5] is None

    def test_pyarrow_divide_by_zero_per_row(self) -> None:
        arrow_table = self._truth_table_arrow()
        fs = self._make_divide_fs()
        result = PyArrowPointArithmetic.calculate_feature(arrow_table, fs)
        values = result.column("a&b__divide_point").to_pylist()
        self._assert_inf_nan_table(values)

    def test_pandas_divide_by_zero_per_row(self) -> None:
        pytest.importorskip("pandas")
        from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.pandas_point_arithmetic import (
            PandasPointArithmetic,
        )

        df = self._truth_table_arrow().to_pandas()
        fs = self._make_divide_fs()
        result = PandasPointArithmetic.calculate_feature(df, fs)
        # ``Series.tolist()`` keeps NaN/inf as Python floats but coerces missing
        # entries to NaN. Distinguish "row had null operand" (we want None) from
        # "row computed to NaN" (we want NaN preserved) by checking the input
        # operands rather than relying on pd.isna on the output.
        raw = result["a&b__divide_point"].tolist()
        col_a = [10, 10, 0, None, 5, -10]
        col_b: list[float | None] = [2.0, 0.0, 0.0, 5.0, None, 0.0]
        values: list[Any] = []
        for i, v in enumerate(raw):
            if col_a[i] is None or col_b[i] is None:
                values.append(None)
            else:
                values.append(v)
        self._assert_inf_nan_table(values)

    def test_polars_divide_by_zero_per_row(self) -> None:
        polars = pytest.importorskip("polars")
        from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.polars_lazy_point_arithmetic import (
            PolarsLazyPointArithmetic,
        )

        arrow_table = self._truth_table_arrow()
        lf = polars.from_arrow(arrow_table).lazy()
        fs = self._make_divide_fs()
        result = PolarsLazyPointArithmetic.calculate_feature(lf, fs)
        collected = result.collect()
        values = collected["a&b__divide_point"].to_list()
        self._assert_inf_nan_table(values)

    def test_duckdb_divide_by_zero_per_row(self) -> None:
        duckdb = pytest.importorskip("duckdb")
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

        from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.duckdb_point_arithmetic import (
            DuckdbPointArithmetic,
        )

        arrow_table = self._truth_table_arrow()
        conn = duckdb.connect(":memory:")
        try:
            relation = DuckdbRelation.from_arrow(conn, arrow_table)
            fs = self._make_divide_fs()
            result = DuckdbPointArithmetic.calculate_feature(relation, fs)
            values = list(result.to_arrow_table().column("a&b__divide_point").to_pylist())
            self._assert_inf_nan_table(values)
        finally:
            conn.close()

    def test_sqlite_divide_by_zero_per_row(self) -> None:
        import sqlite3

        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

        from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.sqlite_point_arithmetic import (
            SqlitePointArithmetic,
        )

        arrow_table = self._truth_table_arrow()
        conn = sqlite3.connect(":memory:")
        try:
            relation = SqliteRelation.from_arrow(conn, arrow_table)
            fs = self._make_divide_fs()
            result = SqlitePointArithmetic.calculate_feature(relation, fs)
            values = list(result.to_arrow_table().column("a&b__divide_point").to_pylist())
            self._assert_null_for_zero_table(values)
        finally:
            conn.close()


class TestMissingInFeatures:
    """Features with only 1 in_features (option-based, no string pattern) must
    raise a ValueError at compute time on every backend."""

    @staticmethod
    def _fs_one_feature() -> FeatureSet:
        return _make_fs("bad_one", op="add", in_features=["value_int"])

    def test_pandas_rejects_one_in_feature(self) -> None:
        pytest.importorskip("pandas")
        from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.pandas_point_arithmetic import (
            PandasPointArithmetic,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        df = arrow_table.to_pandas()
        with pytest.raises(ValueError, match="at least 2"):
            PandasPointArithmetic.calculate_feature(df, self._fs_one_feature())

    def test_pyarrow_rejects_one_in_feature(self) -> None:
        arrow_table = PyArrowDataOpsTestDataCreator.create()
        with pytest.raises(ValueError, match="at least 2"):
            PyArrowPointArithmetic.calculate_feature(arrow_table, self._fs_one_feature())

    def test_polars_rejects_one_in_feature(self) -> None:
        polars = pytest.importorskip("polars")
        from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.polars_lazy_point_arithmetic import (
            PolarsLazyPointArithmetic,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        lf = polars.from_arrow(arrow_table).lazy()
        with pytest.raises(ValueError, match="at least 2"):
            PolarsLazyPointArithmetic.calculate_feature(lf, self._fs_one_feature())

    def test_duckdb_rejects_one_in_feature(self) -> None:
        duckdb = pytest.importorskip("duckdb")
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

        from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.duckdb_point_arithmetic import (
            DuckdbPointArithmetic,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        conn = duckdb.connect(":memory:")
        try:
            relation = DuckdbRelation.from_arrow(conn, arrow_table)
            with pytest.raises(ValueError, match="at least 2"):
                DuckdbPointArithmetic.calculate_feature(relation, self._fs_one_feature())
        finally:
            conn.close()

    def test_sqlite_rejects_one_in_feature(self) -> None:
        import sqlite3

        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

        from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.sqlite_point_arithmetic import (
            SqlitePointArithmetic,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        conn = sqlite3.connect(":memory:")
        try:
            relation = SqliteRelation.from_arrow(conn, arrow_table)
            with pytest.raises(ValueError, match="at least 2"):
                SqlitePointArithmetic.calculate_feature(relation, self._fs_one_feature())
        finally:
            conn.close()


class TestInFeaturesTypeValidation:
    """Verify min/max in_feature counts and missing-column behavior."""

    def test_at_least_2_error_on_single_in_feature(self) -> None:
        arrow_table = PyArrowDataOpsTestDataCreator.create()
        fs = _make_fs("bad", op="add", in_features=["value_int"])
        with pytest.raises(ValueError, match="at least 2"):
            PyArrowPointArithmetic.calculate_feature(arrow_table, fs)

    def test_at_most_2_error_on_three_in_features(self) -> None:
        arrow_table = PyArrowDataOpsTestDataCreator.create()
        fs = _make_fs("bad", op="add", in_features=["value_int", "amount", "value_float"])
        with pytest.raises(ValueError, match="at most 2"):
            PyArrowPointArithmetic.calculate_feature(arrow_table, fs)

    def test_missing_column_named_in_error(self) -> None:
        """An in_feature that names a non-existent column must surface that column name."""
        arrow_table = PyArrowDataOpsTestDataCreator.create()
        fs = _make_fs("bad", op="add", in_features=["value_int", "fake_col"])
        with pytest.raises(ValueError, match=r"fake_col"):
            PyArrowPointArithmetic.calculate_feature(arrow_table, fs)


class TestReservedColumnGuardAllBackends:
    """The ``__mloda_`` reserved-column guard must fire on every backend."""

    @staticmethod
    def _arrow_with_reserved_col() -> Any:
        import pyarrow as pa

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        return arrow_table.append_column("__mloda_rn__", pa.array([0] * 12, type=pa.int64()))

    def test_pandas_rejects_reserved_column(self) -> None:
        pytest.importorskip("pandas")
        from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.pandas_point_arithmetic import (
            PandasPointArithmetic,
        )

        arrow_table = self._arrow_with_reserved_col()
        df = arrow_table.to_pandas()
        fs = _make_fs("value_int&amount__add_point")
        with pytest.raises(ValueError, match=r"(?i)reserved") as exc_info:
            PandasPointArithmetic.calculate_feature(df, fs)
        assert "__mloda_" in str(exc_info.value)

    def test_pyarrow_rejects_reserved_column(self) -> None:
        arrow_table = self._arrow_with_reserved_col()
        fs = _make_fs("value_int&amount__add_point")
        with pytest.raises(ValueError, match=r"(?i)reserved") as exc_info:
            PyArrowPointArithmetic.calculate_feature(arrow_table, fs)
        assert "__mloda_" in str(exc_info.value)

    def test_polars_rejects_reserved_column(self) -> None:
        polars = pytest.importorskip("polars")
        from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.polars_lazy_point_arithmetic import (
            PolarsLazyPointArithmetic,
        )

        arrow_table = self._arrow_with_reserved_col()
        lf = polars.from_arrow(arrow_table).lazy()
        fs = _make_fs("value_int&amount__add_point")
        with pytest.raises(ValueError, match=r"(?i)reserved") as exc_info:
            PolarsLazyPointArithmetic.calculate_feature(lf, fs)
        assert "__mloda_" in str(exc_info.value)

    def test_duckdb_rejects_reserved_column(self) -> None:
        duckdb = pytest.importorskip("duckdb")
        from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.duckdb_point_arithmetic import (
            DuckdbPointArithmetic,
        )

        arrow_table = self._arrow_with_reserved_col()
        conn = duckdb.connect(":memory:")
        try:
            relation = conn.from_arrow(arrow_table)
            fs = _make_fs("value_int&amount__add_point")
            with pytest.raises(ValueError, match=r"(?i)reserved") as exc_info:
                DuckdbPointArithmetic.calculate_feature(relation, fs)
            assert "__mloda_" in str(exc_info.value)
        finally:
            conn.close()

    def test_sqlite_rejects_reserved_column(self) -> None:
        import sqlite3

        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

        from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.sqlite_point_arithmetic import (
            SqlitePointArithmetic,
        )

        arrow_table = self._arrow_with_reserved_col()
        conn = sqlite3.connect(":memory:")
        try:
            relation = SqliteRelation.from_arrow(conn, arrow_table)
            fs = _make_fs("value_int&amount__add_point")
            with pytest.raises(ValueError, match=r"(?i)reserved") as exc_info:
                SqlitePointArithmetic.calculate_feature(relation, fs)
            assert "__mloda_" in str(exc_info.value)
        finally:
            conn.close()


class TestSourceColumnTypeValidation:
    """Both col_a and col_b must be numeric; non-numeric must name the offending column."""

    def test_col_a_non_numeric_named_in_error(self) -> None:
        """When col_a is non-numeric, the error must quote ``name`` (the offender)."""
        import re

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        fs = _make_fs("name&value_int__add_point")
        with pytest.raises(ValueError, match=r"(?i)numeric") as exc_info:
            PyArrowPointArithmetic.calculate_feature(arrow_table, fs)
        assert re.search(r"['\"]name['\"]", str(exc_info.value)), (
            f"Expected offending column 'name' to be quoted, got: {exc_info.value!r}"
        )

    def test_col_b_non_numeric_named_in_error(self) -> None:
        """When col_b is non-numeric, the error must quote ``name`` (the offender)."""
        import re

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        fs = _make_fs("value_int&name__add_point")
        with pytest.raises(ValueError, match=r"(?i)numeric") as exc_info:
            PyArrowPointArithmetic.calculate_feature(arrow_table, fs)
        assert re.search(r"['\"]name['\"]", str(exc_info.value)), (
            f"Expected offending column 'name' to be quoted, got: {exc_info.value!r}"
        )
