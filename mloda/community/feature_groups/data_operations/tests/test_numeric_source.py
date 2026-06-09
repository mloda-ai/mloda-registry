"""Structural single-source-of-truth guards for the per-framework numeric-source modules.

These tests guard the "exactly one place" Definition of Done for issue #214:
the per-backend "what counts as numeric" logic lives in exactly one module per
compute framework (``data_operations/<framework>_numeric_source.py``) and each
such module is wired in by ONE per-backend mixin (under
``data_operations/arithmetic/``) that BOTH the point- and scalar-arithmetic
families inherit.

The target shape they pin:

- The mixins are plain classes at runtime, NOT ``FeatureGroup`` subclasses, so
  they never show up in the FeatureGroup subclass tree / plugin discovery.
- Each mixin contributes only the ``_non_numeric_descriptor`` hook (plus
  ``compute_framework_rule`` / ``_input_columns_and_framework``); the
  ``_assert_source_column_is_numeric`` template lives solely on
  ``ArithmeticFeatureGroupBase`` and nobody overrides it.
- Both concrete classes per backend (point and scalar) bind the mixin's hook
  and the base's template, with the mixin preceding the family base in the MRO.
- The SQL operator map exists once as ``base.SQL_ARITHMETIC_OPS``; the DuckDB
  and SQLite names alias that same object.

They are deliberately structural (``is``-identity of functions and dicts, MRO
order). The actual per-backend numeric behavior on real data is covered by the
shared ``PointArithmeticTestBase`` / ``ScalarArithmeticTestBase`` non-numeric
source rejection tests, which run across all five backends on the canonical
12-row dataset and flow through the per-framework numeric-source modules.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from mloda.provider import FeatureGroup

from mloda.community.feature_groups.data_operations.arithmetic import (
    duckdb_numeric_source,
    pandas_numeric_source,
    polars_numeric_source,
    pyarrow_numeric_source,
    sqlite_numeric_source,
)
from mloda.community.feature_groups.data_operations.arithmetic.base import ArithmeticFeatureGroupBase
from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
    PointArithmeticFeatureGroup,
)
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ScalarArithmeticFeatureGroup,
)

_POINT = "mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic"
_SCALAR = "mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic"
_MIXIN = "mloda.community.feature_groups.data_operations.arithmetic"


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
    """Each backend mixin contributes only the ``_non_numeric_descriptor`` hook; the base owns the template.

    For every backend and BOTH families (point and scalar), the concrete class
    must inherit ``_non_numeric_descriptor`` from the one mixin and
    ``_assert_source_column_is_numeric`` from ``ArithmeticFeatureGroupBase``
    (the template method that calls the hook and raises via
    ``_raise_non_numeric_source``). The mixin must precede the family base in
    the MRO so its classmethods win over any family-level defaults.
    """

    def _assert_concrete_bindings(
        self, mixin_cls: Any, concrete_cls: Any, family_base: type[ArithmeticFeatureGroupBase]
    ) -> None:
        """Pin the mixin-hook / base-template binding contract on one concrete class."""
        base_cls: Any = ArithmeticFeatureGroupBase
        assert concrete_cls._non_numeric_descriptor.__func__ is mixin_cls._non_numeric_descriptor.__func__, (
            f"{concrete_cls.__name__} must inherit _non_numeric_descriptor from {mixin_cls.__name__}"
        )
        assert (
            concrete_cls._assert_source_column_is_numeric.__func__ is base_cls._assert_source_column_is_numeric.__func__
        ), (
            f"{concrete_cls.__name__} must not override the _assert_source_column_is_numeric "
            "template owned by ArithmeticFeatureGroupBase"
        )
        mro = list(concrete_cls.__mro__)
        assert mro.index(mixin_cls) < mro.index(family_base), (
            f"{mixin_cls.__name__} must precede {family_base.__name__} in {concrete_cls.__name__}.__mro__"
        )

    def test_pandas_concrete_classes_bind_mixin_hook(self) -> None:
        """Both pandas concrete classes bind the pandas mixin hook and the base template."""
        pytest.importorskip("pandas")
        mixin_module = importlib.import_module(f"{_MIXIN}.pandas_mixin")
        assert mixin_module.pandas_non_numeric_descriptor is pandas_numeric_source.pandas_non_numeric_descriptor
        point_cls = importlib.import_module(f"{_POINT}.pandas_point_arithmetic").PandasPointArithmetic
        scalar_cls = importlib.import_module(f"{_SCALAR}.pandas_scalar_arithmetic").PandasScalarArithmetic

        mixin_cls = mixin_module.PandasArithmeticMixin
        self._assert_concrete_bindings(mixin_cls, point_cls, PointArithmeticFeatureGroup)
        self._assert_concrete_bindings(mixin_cls, scalar_cls, ScalarArithmeticFeatureGroup)

    def test_polars_concrete_classes_bind_mixin_hook(self) -> None:
        """Both polars concrete classes bind the polars mixin hook and the base template."""
        pytest.importorskip("polars")
        mixin_module = importlib.import_module(f"{_MIXIN}.polars_mixin")
        assert mixin_module.polars_non_numeric_descriptor is polars_numeric_source.polars_non_numeric_descriptor
        point_cls = importlib.import_module(f"{_POINT}.polars_lazy_point_arithmetic").PolarsLazyPointArithmetic
        scalar_cls = importlib.import_module(f"{_SCALAR}.polars_lazy_scalar_arithmetic").PolarsLazyScalarArithmetic

        mixin_cls = mixin_module.PolarsArithmeticMixin
        self._assert_concrete_bindings(mixin_cls, point_cls, PointArithmeticFeatureGroup)
        self._assert_concrete_bindings(mixin_cls, scalar_cls, ScalarArithmeticFeatureGroup)

    def test_pyarrow_concrete_classes_bind_mixin_hook(self) -> None:
        """Both pyarrow concrete classes bind the pyarrow mixin hook and the base template."""
        pytest.importorskip("pyarrow")
        mixin_module = importlib.import_module(f"{_MIXIN}.pyarrow_mixin")
        assert mixin_module.pyarrow_non_numeric_descriptor is pyarrow_numeric_source.pyarrow_non_numeric_descriptor
        point_cls = importlib.import_module(f"{_POINT}.pyarrow_point_arithmetic").PyArrowPointArithmetic
        scalar_cls = importlib.import_module(f"{_SCALAR}.pyarrow_scalar_arithmetic").PyArrowScalarArithmetic

        mixin_cls = mixin_module.PyArrowArithmeticMixin
        self._assert_concrete_bindings(mixin_cls, point_cls, PointArithmeticFeatureGroup)
        self._assert_concrete_bindings(mixin_cls, scalar_cls, ScalarArithmeticFeatureGroup)

    def test_duckdb_concrete_classes_bind_mixin_hook(self) -> None:
        """Both DuckDB concrete classes bind the DuckDB mixin hook and the base template."""
        pytest.importorskip("duckdb")
        mixin_module = importlib.import_module(f"{_MIXIN}.duckdb_mixin")
        assert mixin_module.duckdb_non_numeric_descriptor is duckdb_numeric_source.duckdb_non_numeric_descriptor
        point_cls = importlib.import_module(f"{_POINT}.duckdb_point_arithmetic").DuckdbPointArithmetic
        scalar_cls = importlib.import_module(f"{_SCALAR}.duckdb_scalar_arithmetic").DuckdbScalarArithmetic

        mixin_cls = mixin_module.DuckdbArithmeticMixin
        self._assert_concrete_bindings(mixin_cls, point_cls, PointArithmeticFeatureGroup)
        self._assert_concrete_bindings(mixin_cls, scalar_cls, ScalarArithmeticFeatureGroup)

    def test_sqlite_concrete_classes_bind_mixin_hook(self) -> None:
        """Both SQLite concrete classes bind the SQLite mixin hook and the base template."""
        mixin_module = importlib.import_module(f"{_MIXIN}.sqlite_mixin")
        assert mixin_module.sqlite_non_numeric_descriptor is sqlite_numeric_source.sqlite_non_numeric_descriptor
        point_cls = importlib.import_module(f"{_POINT}.sqlite_point_arithmetic").SqlitePointArithmetic
        scalar_cls = importlib.import_module(f"{_SCALAR}.sqlite_scalar_arithmetic").SqliteScalarArithmetic

        mixin_cls = mixin_module.SqliteArithmeticMixin
        self._assert_concrete_bindings(mixin_cls, point_cls, PointArithmeticFeatureGroup)
        self._assert_concrete_bindings(mixin_cls, scalar_cls, ScalarArithmeticFeatureGroup)


class TestMixinsAreNotFeatureGroups:
    """The arithmetic mixins are plain runtime classes, invisible to FeatureGroup plugin discovery.

    The mixins must not subclass ``FeatureGroup`` at runtime (the shared base
    may be a TYPE_CHECKING-only parent), so walking the FeatureGroup subclass
    tree, which plugin discovery does, never yields a half-abstract mixin.
    """

    def _assert_plain_class(self, mixin_cls: Any) -> None:
        """Assert the mixin is outside the FeatureGroup subclass tree at runtime."""
        assert not issubclass(mixin_cls, FeatureGroup), (
            f"{mixin_cls.__name__} must not be a runtime FeatureGroup subclass; "
            "it would leak into the FeatureGroup subclass tree and plugin discovery."
        )

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
    """The SQL operator map lives once in ``arithmetic.base`` and the SQL backends alias it.

    DuckDB and SQLite previously each declared a byte-identical operator dict.
    The shared map is ``base.SQL_ARITHMETIC_OPS``; the per-backend names
    (``DUCKDB_ARITHMETIC_OPS`` / ``SQLITE_ARITHMETIC_OPS``) must be the SAME
    object, not copies, so the dicts can never drift apart.
    """

    def test_sql_arithmetic_ops_content(self) -> None:
        """``base.SQL_ARITHMETIC_OPS`` maps the four shared op names to SQL operators."""
        base_module = importlib.import_module(f"{_MIXIN}.base")
        assert base_module.SQL_ARITHMETIC_OPS == {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}

    def test_duckdb_arithmetic_ops_is_shared_sql_ops(self) -> None:
        """``DUCKDB_ARITHMETIC_OPS`` is the very object declared in ``arithmetic.base``."""
        pytest.importorskip("duckdb")
        base_module = importlib.import_module(f"{_MIXIN}.base")
        mixin_module = importlib.import_module(f"{_MIXIN}.duckdb_mixin")
        assert mixin_module.DUCKDB_ARITHMETIC_OPS is base_module.SQL_ARITHMETIC_OPS

    def test_sqlite_arithmetic_ops_is_shared_sql_ops(self) -> None:
        """``SQLITE_ARITHMETIC_OPS`` is the very object declared in ``arithmetic.base``."""
        base_module = importlib.import_module(f"{_MIXIN}.base")
        mixin_module = importlib.import_module(f"{_MIXIN}.sqlite_mixin")
        assert mixin_module.SQLITE_ARITHMETIC_OPS is base_module.SQL_ARITHMETIC_OPS
