"""Scalar-arithmetic-specific security tests: compute-level rejection, backend
allowlist completeness, divide-by-zero handling, and missing-constant handling.

Generic match-validation tests live in ``test_base.py`` via
``MatchValidationTestBase``.
"""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ARITHMETIC_OPERATIONS,
)
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.pyarrow_scalar_arithmetic import (
    PyArrowScalarArithmetic,
)


def _make_fs(name: str, *, constant: int | float | None = None, op: str | None = None) -> FeatureSet:
    """Build a FeatureSet for a scalar arithmetic feature.

    If ``op`` is given, set arithmetic_op + in_features context (option-based).
    If ``constant`` is None, omit it (used to test missing-constant rejection).
    """
    context: dict[str, Any] = {}
    if constant is not None:
        context["constant"] = constant
    if op is not None:
        context["arithmetic_op"] = op
        context["in_features"] = "value_int"
    feature = Feature(name, options=Options(context=context))
    fs = FeatureSet()
    fs.add(feature)
    return fs


class TestScalarArithmeticComputeRejection:
    """Verify that invalid ops reaching compute raise ValueError."""

    def test_pandas_rejects_unknown_op_at_compute(self) -> None:
        pytest.importorskip("pandas")
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.pandas_scalar_arithmetic import (
            PandasScalarArithmetic,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        df = arrow_table.to_pandas()
        with pytest.raises(ValueError, match="[Uu]nsupported"):
            PandasScalarArithmetic._compute_arithmetic(df, "result_col", "value_int", "evil_op", 5)

    def test_pyarrow_rejects_unknown_op_at_compute(self) -> None:
        arrow_table = PyArrowDataOpsTestDataCreator.create()
        with pytest.raises(ValueError, match="[Uu]nsupported"):
            PyArrowScalarArithmetic._compute_arithmetic(arrow_table, "result_col", "value_int", "evil_op", 5)


class TestAllowlistCompleteness:
    """Verify that every op in ARITHMETIC_OPERATIONS is covered by every backend."""

    def test_pandas_covers_all_ops(self) -> None:
        pytest.importorskip("pandas")
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.pandas_scalar_arithmetic import (
            PandasScalarArithmetic,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        df = arrow_table.to_pandas()
        for op in ARITHMETIC_OPERATIONS:
            constant: int | float = 2.0 if op == "divide" else 1
            result = PandasScalarArithmetic._compute_arithmetic(df, f"test_{op}", "value_int", op, constant)
            assert f"test_{op}" in result.columns

    def test_pyarrow_covers_all_ops(self) -> None:
        arrow_table = PyArrowDataOpsTestDataCreator.create()
        for op in ARITHMETIC_OPERATIONS:
            constant: int | float = 2.0 if op == "divide" else 1
            result = PyArrowScalarArithmetic._compute_arithmetic(arrow_table, f"test_{op}", "value_int", op, constant)
            assert f"test_{op}" in result.column_names

    def test_polars_covers_all_ops(self) -> None:
        polars = pytest.importorskip("polars")
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.polars_lazy_scalar_arithmetic import (
            PolarsLazyScalarArithmetic,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        lf = polars.from_arrow(arrow_table).lazy()
        for op in ARITHMETIC_OPERATIONS:
            constant: int | float = 2.0 if op == "divide" else 1
            result = PolarsLazyScalarArithmetic._compute_arithmetic(lf, f"test_{op}", "value_int", op, constant)
            collected = result.collect()
            assert f"test_{op}" in collected.columns

    def test_duckdb_covers_all_ops(self) -> None:
        pytest.importorskip("duckdb")
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.duckdb_scalar_arithmetic import (
            DUCKDB_ARITHMETIC_OPS,
        )

        for op in ARITHMETIC_OPERATIONS:
            assert op in DUCKDB_ARITHMETIC_OPS, f"DuckDB backend missing arithmetic op: {op}"

    def test_sqlite_covers_all_ops(self) -> None:
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.sqlite_scalar_arithmetic import (
            SQLITE_ARITHMETIC_OPS,
        )

        for op in ARITHMETIC_OPERATIONS:
            assert op in SQLITE_ARITHMETIC_OPS, f"SQLite backend missing arithmetic op: {op}"


class TestDivideByZero:
    """Divide-by-zero must raise ValueError on every backend."""

    def test_pandas_divide_by_zero_raises(self) -> None:
        pytest.importorskip("pandas")
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.pandas_scalar_arithmetic import (
            PandasScalarArithmetic,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        df = arrow_table.to_pandas()
        fs = _make_fs("value_int__divide_constant", constant=0)
        with pytest.raises(ValueError, match="[Dd]ivide|[Zz]ero|0"):
            PandasScalarArithmetic.calculate_feature(df, fs)

    def test_pyarrow_divide_by_zero_raises(self) -> None:
        arrow_table = PyArrowDataOpsTestDataCreator.create()
        fs = _make_fs("value_int__divide_constant", constant=0)
        with pytest.raises(ValueError, match="[Dd]ivide|[Zz]ero|0"):
            PyArrowScalarArithmetic.calculate_feature(arrow_table, fs)

    def test_polars_divide_by_zero_raises(self) -> None:
        polars = pytest.importorskip("polars")
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.polars_lazy_scalar_arithmetic import (
            PolarsLazyScalarArithmetic,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        lf = polars.from_arrow(arrow_table).lazy()
        fs = _make_fs("value_int__divide_constant", constant=0)
        with pytest.raises(ValueError, match="[Dd]ivide|[Zz]ero|0"):
            PolarsLazyScalarArithmetic.calculate_feature(lf, fs)

    def test_duckdb_divide_by_zero_raises(self) -> None:
        duckdb = pytest.importorskip("duckdb")
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.duckdb_scalar_arithmetic import (
            DuckdbScalarArithmetic,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        conn = duckdb.connect(":memory:")
        try:
            relation = conn.from_arrow(arrow_table)
            fs = _make_fs("value_int__divide_constant", constant=0)
            with pytest.raises(ValueError, match="[Dd]ivide|[Zz]ero|0"):
                DuckdbScalarArithmetic.calculate_feature(relation, fs)
        finally:
            conn.close()

    def test_sqlite_divide_by_zero_raises(self) -> None:
        import sqlite3

        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

        from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.sqlite_scalar_arithmetic import (
            SqliteScalarArithmetic,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        conn = sqlite3.connect(":memory:")
        try:
            relation = SqliteRelation.from_arrow(conn, arrow_table)
            fs = _make_fs("value_int__divide_constant", constant=0)
            with pytest.raises(ValueError, match="[Dd]ivide|[Zz]ero|0"):
                SqliteScalarArithmetic.calculate_feature(relation, fs)
        finally:
            conn.close()


class TestMissingConstant:
    """Missing constant in Options must raise ValueError on every backend."""

    def test_pandas_missing_constant_raises(self) -> None:
        pytest.importorskip("pandas")
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.pandas_scalar_arithmetic import (
            PandasScalarArithmetic,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        df = arrow_table.to_pandas()
        fs = _make_fs("value_int__add_constant")
        with pytest.raises(ValueError, match="constant"):
            PandasScalarArithmetic.calculate_feature(df, fs)

    def test_pyarrow_missing_constant_raises(self) -> None:
        arrow_table = PyArrowDataOpsTestDataCreator.create()
        fs = _make_fs("value_int__add_constant")
        with pytest.raises(ValueError, match="constant"):
            PyArrowScalarArithmetic.calculate_feature(arrow_table, fs)

    def test_polars_missing_constant_raises(self) -> None:
        polars = pytest.importorskip("polars")
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.polars_lazy_scalar_arithmetic import (
            PolarsLazyScalarArithmetic,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        lf = polars.from_arrow(arrow_table).lazy()
        fs = _make_fs("value_int__add_constant")
        with pytest.raises(ValueError, match="constant"):
            PolarsLazyScalarArithmetic.calculate_feature(lf, fs)

    def test_duckdb_missing_constant_raises(self) -> None:
        duckdb = pytest.importorskip("duckdb")
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.duckdb_scalar_arithmetic import (
            DuckdbScalarArithmetic,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        conn = duckdb.connect(":memory:")
        try:
            relation = conn.from_arrow(arrow_table)
            fs = _make_fs("value_int__add_constant")
            with pytest.raises(ValueError, match="constant"):
                DuckdbScalarArithmetic.calculate_feature(relation, fs)
        finally:
            conn.close()

    def test_sqlite_missing_constant_raises(self) -> None:
        import sqlite3

        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

        from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.sqlite_scalar_arithmetic import (
            SqliteScalarArithmetic,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        conn = sqlite3.connect(":memory:")
        try:
            relation = SqliteRelation.from_arrow(conn, arrow_table)
            fs = _make_fs("value_int__add_constant")
            with pytest.raises(ValueError, match="constant"):
                SqliteScalarArithmetic.calculate_feature(relation, fs)
        finally:
            conn.close()
