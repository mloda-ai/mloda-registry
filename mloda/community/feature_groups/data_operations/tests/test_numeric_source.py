"""Structural single-source-of-truth guards for the per-framework numeric-source modules.

Each framework's "what counts as numeric" logic lives in one module
(``data_operations/<framework>_numeric_source.py``), wired in by ONE plain mixin per
framework (under ``data_operations/row_preserving/arithmetic/``) that carries only
``_non_numeric_descriptor``. Both the point and scalar Implementations import that mixin
and pass its hook to the group's ``apply_arithmetic``; the groups and Implementations
define no numeric-source hook. The SQL operator map exists once as
``base.SQL_ARITHMETIC_OPS``, imported by the DuckDB and SQLite Implementations.

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
_MIXIN = "mloda.community.feature_groups.data_operations.row_preserving.arithmetic"


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


class TestMixinHookBindings:
    """Each backend module imports its framework's one mixin; the Implementation defines no numeric-source hook.

    For every backend and BOTH families (point and scalar), the backend module
    imports the one mixin of its framework (whose ``_non_numeric_descriptor`` hook
    is the numeric-source module's function) and passes that hook to the family
    group's ``apply_arithmetic``. The Implementation itself defines none of
    ``_non_numeric_descriptor``, ``_assert_source_column_is_numeric`` or
    ``compute_framework_rule``, the mixin defines no ``compute_framework_rule``, and
    neither group defines or inherits the numeric-source hooks.
    """

    _HOOK_NAMES = ("_non_numeric_descriptor", "_assert_source_column_is_numeric", "compute_framework_rule")

    def _assert_module_binds_mixin(self, mixin_cls: Any, module_name: str, concrete_name: str) -> None:
        """Pin the module-imports-mixin / Implementation-defines-no-hook contract on one backend module."""
        module = importlib.import_module(module_name)
        assert getattr(module, mixin_cls.__name__) is mixin_cls, (
            f"{module_name} must import {mixin_cls.__name__} from the shared arithmetic package"
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

    def _assert_mixin_defines_no_compute_framework_rule(self, mixin_cls: Any) -> None:
        assert "compute_framework_rule" not in vars(mixin_cls), (
            f"{mixin_cls.__name__} must not define compute_framework_rule"
        )

    def _assert_both_families(self, mixin_cls: Any, backend: str, class_prefix: str) -> None:
        self._assert_module_binds_mixin(
            mixin_cls, f"{_POINT}.{backend}_point_arithmetic", f"{class_prefix}PointArithmetic"
        )
        self._assert_module_binds_mixin(
            mixin_cls, f"{_SCALAR}.{backend}_scalar_arithmetic", f"{class_prefix}ScalarArithmetic"
        )
        self._assert_groups_have_no_numeric_source_hook()
        self._assert_mixin_defines_no_compute_framework_rule(mixin_cls)

    def test_pandas_modules_bind_mixin_hook(self) -> None:
        """Both pandas backend modules import the pandas mixin."""
        pytest.importorskip("pandas")
        mixin_module = importlib.import_module(f"{_MIXIN}.pandas_mixin")
        assert mixin_module.pandas_non_numeric_descriptor is pandas_numeric_source.pandas_non_numeric_descriptor
        self._assert_both_families(mixin_module.PandasArithmeticMixin, "pandas", "Pandas")

    def test_polars_modules_bind_mixin_hook(self) -> None:
        """Both polars backend modules import the polars mixin."""
        pytest.importorskip("polars")
        mixin_module = importlib.import_module(f"{_MIXIN}.polars_mixin")
        assert mixin_module.polars_non_numeric_descriptor is polars_numeric_source.polars_non_numeric_descriptor
        self._assert_both_families(mixin_module.PolarsArithmeticMixin, "polars_lazy", "PolarsLazy")

    def test_pyarrow_modules_bind_mixin_hook(self) -> None:
        """Both pyarrow backend modules import the pyarrow mixin."""
        pytest.importorskip("pyarrow")
        mixin_module = importlib.import_module(f"{_MIXIN}.pyarrow_mixin")
        assert mixin_module.pyarrow_non_numeric_descriptor is pyarrow_numeric_source.pyarrow_non_numeric_descriptor
        self._assert_both_families(mixin_module.PyArrowArithmeticMixin, "pyarrow", "PyArrow")

    def test_duckdb_modules_bind_mixin_hook(self) -> None:
        """Both DuckDB backend modules import the DuckDB mixin."""
        pytest.importorskip("duckdb")
        mixin_module = importlib.import_module(f"{_MIXIN}.duckdb_mixin")
        assert mixin_module.duckdb_non_numeric_descriptor is duckdb_numeric_source.duckdb_non_numeric_descriptor
        self._assert_both_families(mixin_module.DuckdbArithmeticMixin, "duckdb", "Duckdb")

    def test_sqlite_modules_bind_mixin_hook(self) -> None:
        """Both SQLite backend modules import the SQLite mixin."""
        mixin_module = importlib.import_module(f"{_MIXIN}.sqlite_mixin")
        assert mixin_module.sqlite_non_numeric_descriptor is sqlite_numeric_source.sqlite_non_numeric_descriptor
        self._assert_both_families(mixin_module.SqliteArithmeticMixin, "sqlite", "Sqlite")


class TestMixinsAreNotFeatureGroups:
    """The arithmetic mixins are not FeatureGroups, so plugin discovery never yields one."""

    def _assert_plain_class(self, mixin_cls: Any) -> None:
        """Assert the mixin is outside the FeatureGroup subclass tree at runtime."""
        assert not issubclass(mixin_cls, FeatureGroup), (
            f"{mixin_cls.__name__} must not be a runtime FeatureGroup subclass; "
            "it would leak into the FeatureGroup subclass tree and plugin discovery."
        )

    def test_arithmetic_op_mixin_is_not_a_feature_group(self) -> None:
        """The shared ``ArithmeticOpMixin`` stays out of the FeatureGroup subclass tree."""
        self._assert_plain_class(ArithmeticOpMixin)

    def test_pandas_mixin_is_not_a_feature_group(self) -> None:
        """``PandasArithmeticMixin`` stays out of the FeatureGroup subclass tree."""
        pytest.importorskip("pandas")
        mixin_module = importlib.import_module(f"{_MIXIN}.pandas_mixin")
        self._assert_plain_class(mixin_module.PandasArithmeticMixin)

    def test_polars_mixin_is_not_a_feature_group(self) -> None:
        """``PolarsArithmeticMixin`` stays out of the FeatureGroup subclass tree."""
        pytest.importorskip("polars")
        mixin_module = importlib.import_module(f"{_MIXIN}.polars_mixin")
        self._assert_plain_class(mixin_module.PolarsArithmeticMixin)

    def test_pyarrow_mixin_is_not_a_feature_group(self) -> None:
        """``PyArrowArithmeticMixin`` stays out of the FeatureGroup subclass tree."""
        pytest.importorskip("pyarrow")
        mixin_module = importlib.import_module(f"{_MIXIN}.pyarrow_mixin")
        self._assert_plain_class(mixin_module.PyArrowArithmeticMixin)

    def test_duckdb_mixin_is_not_a_feature_group(self) -> None:
        """``DuckdbArithmeticMixin`` stays out of the FeatureGroup subclass tree."""
        pytest.importorskip("duckdb")
        mixin_module = importlib.import_module(f"{_MIXIN}.duckdb_mixin")
        self._assert_plain_class(mixin_module.DuckdbArithmeticMixin)

    def test_sqlite_mixin_is_not_a_feature_group(self) -> None:
        """``SqliteArithmeticMixin`` stays out of the FeatureGroup subclass tree."""
        mixin_module = importlib.import_module(f"{_MIXIN}.sqlite_mixin")
        self._assert_plain_class(mixin_module.SqliteArithmeticMixin)


class TestSqlArithmeticOpsSingleSourceOfTruth:
    """The SQL operator map lives once in ``arithmetic.base`` and the SQL Implementations import it."""

    def test_sql_arithmetic_ops_content(self) -> None:
        """``base.SQL_ARITHMETIC_OPS`` maps the four shared op names to SQL operators."""
        base_module = importlib.import_module(f"{_MIXIN}.base")
        assert base_module.SQL_ARITHMETIC_OPS == {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}

    def test_duckdb_implementations_use_shared_sql_ops(self) -> None:
        """The DuckDB point and scalar modules expose the very ``SQL_ARITHMETIC_OPS`` of ``arithmetic.base``."""
        pytest.importorskip("duckdb")
        base_module = importlib.import_module(f"{_MIXIN}.base")
        for path in (f"{_POINT}.duckdb_point_arithmetic", f"{_SCALAR}.duckdb_scalar_arithmetic"):
            module = importlib.import_module(path)
            assert module.SQL_ARITHMETIC_OPS is base_module.SQL_ARITHMETIC_OPS

    def test_sqlite_implementations_use_shared_sql_ops(self) -> None:
        """The SQLite point and scalar modules expose the very ``SQL_ARITHMETIC_OPS`` of ``arithmetic.base``."""
        base_module = importlib.import_module(f"{_MIXIN}.base")
        for path in (f"{_POINT}.sqlite_point_arithmetic", f"{_SCALAR}.sqlite_scalar_arithmetic"):
            module = importlib.import_module(path)
            assert module.SQL_ARITHMETIC_OPS is base_module.SQL_ARITHMETIC_OPS
