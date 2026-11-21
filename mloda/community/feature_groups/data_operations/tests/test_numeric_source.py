"""Structural single-source-of-truth guards for the per-framework numeric-source modules.

Each framework's numeric check lives in one module,
``row_preserving/arithmetic/<framework>_numeric_source.py``, whose
``<framework>_non_numeric_descriptor(data, source_col)`` both the point and scalar
Implementations pass to ``apply_arithmetic``; the groups and Implementations define no
numeric-source hook. The SQL operator map exists once as ``base.SQL_ARITHMETIC_OPS``,
imported by the DuckDB and SQLite Implementations.

The tests are structural (``is``-identity of functions and dicts, class attributes); the
numeric behavior on real data is covered by the shared non-numeric source rejection tests.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest
from mloda.provider import FeatureGroup

from mloda.community.feature_groups.data_operations.row_preserving.arithmetic import (
    duckdb_numeric_source,
    pandas_numeric_source,
    polars_numeric_source,
    pyarrow_numeric_source,
    sqlite_numeric_source,
)
from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.base import ArithmeticOpMixin
from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
    PointArithmeticFeatureGroup,
)
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ScalarArithmeticFeatureGroup,
)

_POINT = "mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic"
_SCALAR = "mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic"
_ARITH = "mloda.community.feature_groups.data_operations.row_preserving.arithmetic"


class TestNumericSourceSingleSourceOfTruth:
    """Guard that each framework's numeric-source logic lives in one module, imported by both families."""

    def test_duckdb_numeric_prefixes_content(self) -> None:
        """The DuckDB numeric allowlist is exposed by the shared module with the expected names."""
        prefixes = duckdb_numeric_source.DUCKDB_NUMERIC_PREFIXES
        for expected in ("TINYINT", "BIGINT", "HUGEINT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "BIGNUM"):
            assert expected in prefixes, f"{expected!r} missing from duckdb_numeric_source.DUCKDB_NUMERIC_PREFIXES"

    def test_duckdb_backend_modules_do_not_redeclare_prefixes(self) -> None:
        """Neither DuckDB backend may carry its own copy of the numeric allowlist."""
        pytest.importorskip("duckdb")
        for path in (
            f"{_POINT}.duckdb_point_arithmetic",
            f"{_SCALAR}.duckdb_scalar_arithmetic",
        ):
            module = importlib.import_module(path)
            assert not hasattr(module, "_DUCKDB_NUMERIC_PREFIXES"), (
                f"{path} still declares a module-level _DUCKDB_NUMERIC_PREFIXES; "
                "it must be removed in favour of duckdb_numeric_source.DUCKDB_NUMERIC_PREFIXES."
            )


class TestNumericSourceBindings:
    """Each backend module imports its framework's descriptor; the Implementation defines no numeric-source hook.

    For every backend and BOTH families (point and scalar), the backend module's
    ``<fw>_non_numeric_descriptor`` is the function of ``<fw>_numeric_source``. The
    Implementation defines or inherits none of ``_non_numeric_descriptor``,
    ``_assert_source_column_is_numeric`` or ``compute_framework_rule``, and neither group
    defines or inherits the numeric-source hooks.
    """

    _HOOK_NAMES = ("_non_numeric_descriptor", "_assert_source_column_is_numeric", "compute_framework_rule")

    def _assert_module_binds_descriptor(
        self, numeric_source: Any, framework: str, module_name: str, concrete_name: str
    ) -> None:
        """Pin the module-imports-descriptor / Implementation-defines-no-hook contract on one backend module."""
        name = f"{framework}_non_numeric_descriptor"
        module = importlib.import_module(module_name)
        assert getattr(module, name, None) is getattr(numeric_source, name), (
            f"{module_name} must import {name} from {framework}_numeric_source"
        )
        concrete_cls = getattr(module, concrete_name)
        for hook in self._HOOK_NAMES:
            assert hook not in vars(concrete_cls), f"{concrete_name} must not define {hook}"
            assert not hasattr(concrete_cls, hook), f"{concrete_name} must not inherit {hook}"

    def _assert_groups_have_no_numeric_source_hook(self) -> None:
        for group in (PointArithmeticFeatureGroup, ScalarArithmeticFeatureGroup):
            assert issubclass(group, ArithmeticOpMixin)
            for hook in ("_non_numeric_descriptor", "_assert_source_column_is_numeric"):
                assert not hasattr(group, hook), f"{group.__name__} must neither define nor inherit {hook}"

    def _assert_both_families(self, numeric_source: Any, framework: str, backend: str, class_prefix: str) -> None:
        self._assert_module_binds_descriptor(
            numeric_source, framework, f"{_POINT}.{backend}_point_arithmetic", f"{class_prefix}PointArithmetic"
        )
        self._assert_module_binds_descriptor(
            numeric_source, framework, f"{_SCALAR}.{backend}_scalar_arithmetic", f"{class_prefix}ScalarArithmetic"
        )
        self._assert_groups_have_no_numeric_source_hook()

    def test_pandas_modules_bind_descriptor(self) -> None:
        """Both pandas backend modules import the pandas descriptor."""
        pytest.importorskip("pandas")
        self._assert_both_families(pandas_numeric_source, "pandas", "pandas", "Pandas")

    def test_polars_modules_bind_descriptor(self) -> None:
        """Both polars backend modules import the polars descriptor."""
        pytest.importorskip("polars")
        self._assert_both_families(polars_numeric_source, "polars", "polars_lazy", "PolarsLazy")

    def test_pyarrow_modules_bind_descriptor(self) -> None:
        """Both pyarrow backend modules import the pyarrow descriptor."""
        pytest.importorskip("pyarrow")
        self._assert_both_families(pyarrow_numeric_source, "pyarrow", "pyarrow", "PyArrow")

    def test_duckdb_modules_bind_descriptor(self) -> None:
        """Both DuckDB backend modules import the DuckDB descriptor."""
        pytest.importorskip("duckdb")
        self._assert_both_families(duckdb_numeric_source, "duckdb", "duckdb", "Duckdb")

    def test_sqlite_modules_bind_descriptor(self) -> None:
        """Both SQLite backend modules import the SQLite descriptor."""
        self._assert_both_families(sqlite_numeric_source, "sqlite", "sqlite", "Sqlite")


class TestArithmeticOpMixinIsNotAFeatureGroup:
    """The shared arithmetic mixin is not a FeatureGroup, so plugin discovery never yields it."""

    def test_arithmetic_op_mixin_is_not_a_feature_group(self) -> None:
        """The shared ``ArithmeticOpMixin`` stays out of the FeatureGroup subclass tree."""
        assert not issubclass(ArithmeticOpMixin, FeatureGroup), (
            "ArithmeticOpMixin must not be a runtime FeatureGroup subclass; "
            "it would leak into the FeatureGroup subclass tree and plugin discovery."
        )


class TestSqlArithmeticOpsSingleSourceOfTruth:
    """The SQL operator map lives once in ``arithmetic.base`` and the SQL Implementations import it."""

    def test_sql_arithmetic_ops_content(self) -> None:
        """``base.SQL_ARITHMETIC_OPS`` maps the four shared op names to SQL operators."""
        base_module = importlib.import_module(f"{_ARITH}.base")
        assert base_module.SQL_ARITHMETIC_OPS == {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}

    def test_duckdb_implementations_use_shared_sql_ops(self) -> None:
        """The DuckDB point and scalar modules expose the very ``SQL_ARITHMETIC_OPS`` of ``arithmetic.base``."""
        pytest.importorskip("duckdb")
        base_module = importlib.import_module(f"{_ARITH}.base")
        for path in (f"{_POINT}.duckdb_point_arithmetic", f"{_SCALAR}.duckdb_scalar_arithmetic"):
            module = importlib.import_module(path)
            assert module.SQL_ARITHMETIC_OPS is base_module.SQL_ARITHMETIC_OPS

    def test_sqlite_implementations_use_shared_sql_ops(self) -> None:
        """The SQLite point and scalar modules expose the very ``SQL_ARITHMETIC_OPS`` of ``arithmetic.base``."""
        base_module = importlib.import_module(f"{_ARITH}.base")
        for path in (f"{_POINT}.sqlite_point_arithmetic", f"{_SCALAR}.sqlite_scalar_arithmetic"):
            module = importlib.import_module(path)
            assert module.SQL_ARITHMETIC_OPS is base_module.SQL_ARITHMETIC_OPS
