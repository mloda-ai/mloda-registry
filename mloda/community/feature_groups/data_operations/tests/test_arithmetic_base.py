"""Structural guards for the arithmetic families: one shared plain mixin and the operation-group shape."""

from __future__ import annotations

from typing import Any

import pytest
from mloda.provider import FeatureChainParserMixin, FeatureGroup, OperationFeatureGroup
from mloda.user import Options

from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.base import (
    ARITHMETIC_OP_NAMES,
    ArithmeticOpMixin,
)
from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
    ARITHMETIC_OPERATIONS as POINT_ARITHMETIC_OPERATIONS,
)
from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
    PointArithmeticFeatureGroup,
)
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ARITHMETIC_OPERATIONS as SCALAR_ARITHMETIC_OPERATIONS,
)
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ScalarArithmeticFeatureGroup,
)
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.testing.feature_groups.data_operations.operation_group import ImplSpec, OperationGroupTestBase


class TestArithmeticOpMixin:
    """Both arithmetic families mix in the shared plain ``ArithmeticOpMixin``."""

    def test_point_arithmetic_subclasses_mixin(self) -> None:
        """``PointArithmeticFeatureGroup`` inherits from the shared mixin."""
        assert issubclass(PointArithmeticFeatureGroup, ArithmeticOpMixin)

    def test_scalar_arithmetic_subclasses_mixin(self) -> None:
        """``ScalarArithmeticFeatureGroup`` inherits from the shared mixin."""
        assert issubclass(ScalarArithmeticFeatureGroup, ArithmeticOpMixin)

    @pytest.mark.parametrize(
        "group", [PointArithmeticFeatureGroup, ScalarArithmeticFeatureGroup], ids=lambda g: g.__name__
    )
    def test_group_mro_puts_the_mixins_before_the_operation_group(self, group: Any) -> None:
        """Each arithmetic family lists the mixins before ``OperationFeatureGroup``."""
        assert group.__mro__[1:5] == (
            ArithmeticOpMixin,
            FeatureChainParserMixin,
            OperationFeatureGroup,
            FeatureGroup,
        )


class TestArithmeticOpNamesSingleSourceOfTruth:
    """Guard that each family's op set stays in lockstep with the shared ``ARITHMETIC_OP_NAMES``.

    Each family keeps its own ``ARITHMETIC_OPERATIONS`` dict whose KEYS are the
    op names (mapped to family-specific descriptions), but the matcher accepts
    an op only if it is in the module-level ``ARITHMETIC_OP_NAMES`` frozenset in
    ``arithmetic.base`` (used by ``_validate_string_match``).
    If a family added an op to its dict without it being in the shared frozenset
    (or vice versa), the op would be silently rejected at match time: exactly
    the divergence class issue #214 targets. These structural guards pin the
    key sets to the single source of truth and pin each family's non-empty
    ``OPERATION_LABEL`` (used in the numeric-source rejection message).
    """

    def test_point_arithmetic_operation_keys_match_shared_op_names(self) -> None:
        """The point family's ``ARITHMETIC_OPERATIONS`` keys equal the shared op names."""
        assert set(POINT_ARITHMETIC_OPERATIONS) == ARITHMETIC_OP_NAMES

    def test_scalar_arithmetic_operation_keys_match_shared_op_names(self) -> None:
        """The scalar family's ``ARITHMETIC_OPERATIONS`` keys equal the shared op names."""
        assert set(SCALAR_ARITHMETIC_OPERATIONS) == ARITHMETIC_OP_NAMES

    def test_point_arithmetic_operation_label(self) -> None:
        """``PointArithmeticFeatureGroup`` sets a non-empty, family-specific ``OPERATION_LABEL``."""
        assert PointArithmeticFeatureGroup.OPERATION_LABEL == "point arithmetic"
        assert PointArithmeticFeatureGroup.OPERATION_LABEL

    def test_scalar_arithmetic_operation_label(self) -> None:
        """``ScalarArithmeticFeatureGroup`` sets a non-empty, family-specific ``OPERATION_LABEL``."""
        assert ScalarArithmeticFeatureGroup.OPERATION_LABEL == "scalar arithmetic"
        assert ScalarArithmeticFeatureGroup.OPERATION_LABEL


_FRAMEWORKS = {
    "pyarrow": ("mloda_plugins.compute_framework.base_implementations.pyarrow.table", "PyArrowTable"),
    "pandas": ("mloda_plugins.compute_framework.base_implementations.pandas.dataframe", "PandasDataFrame"),
    "polars_lazy": (
        "mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe",
        "PolarsLazyDataFrame",
    ),
    "duckdb": ("mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework", "DuckDBFramework"),
    "sqlite": ("mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework", "SqliteFramework"),
    "python_dict": (
        "mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework",
        "PythonDictFramework",
    ),
}
_CLASS_NAMES = {
    "pyarrow": "PyArrow",
    "pandas": "Pandas",
    "polars_lazy": "PolarsLazy",
    "duckdb": "Duckdb",
    "sqlite": "Sqlite",
    "python_dict": "PythonDict",
}
_OPTIONAL_DEPENDENCIES = {"pandas": "pandas", "polars_lazy": "polars", "duckdb": "duckdb"}
_FAMILY_PACKAGE = "mloda.community.feature_groups.data_operations.row_preserving"


def _implementation_specs(family: str) -> list[ImplSpec]:
    """The family's Implementations; only point arithmetic on SQLite diverges, on ``divide``."""
    return [
        ImplSpec(
            backend,
            f"{backend}_{family}_arithmetic",
            f"{_CLASS_NAMES[backend]}{family.capitalize()}Arithmetic",
            _FRAMEWORKS[backend][0],
            _FRAMEWORKS[backend][1],
            divergent=frozenset({"divide"}) if (family, backend) == ("point", "sqlite") else frozenset(),
            dependency=_OPTIONAL_DEPENDENCIES.get(backend),
        )
        for backend in _FRAMEWORKS
    ]


class TestPointArithmeticOperationGroup(OperationGroupTestBase):
    @classmethod
    def group_class(cls) -> Any:
        return PointArithmeticFeatureGroup

    @classmethod
    def backend_package(cls) -> str:
        return f"{_FAMILY_PACKAGE}.point_arithmetic"

    @classmethod
    def implementations(cls) -> list[ImplSpec]:
        return _implementation_specs("point")

    @classmethod
    def subtype_key(cls) -> str:
        return "arithmetic_op"

    @classmethod
    def loop_method(cls) -> str:
        return "apply_arithmetic"

    @classmethod
    def compute_method(cls) -> str:
        return "_compute_arithmetic"

    @classmethod
    def probe_feature_set(cls) -> Any:
        return make_feature_set("a&b__add_point")

    @classmethod
    def name_probes(cls) -> list[tuple[str, str]]:
        return [("a&b__add_point", "add")]

    @classmethod
    def option_probes(cls) -> list[tuple[str, Options, str]]:
        return [("my_result", Options(context={"arithmetic_op": "add", "in_features": ["a", "b"]}), "add")]


class TestScalarArithmeticOperationGroup(OperationGroupTestBase):
    @classmethod
    def group_class(cls) -> Any:
        return ScalarArithmeticFeatureGroup

    @classmethod
    def backend_package(cls) -> str:
        return f"{_FAMILY_PACKAGE}.scalar_arithmetic"

    @classmethod
    def implementations(cls) -> list[ImplSpec]:
        return _implementation_specs("scalar")

    @classmethod
    def subtype_key(cls) -> str:
        return "arithmetic_op"

    @classmethod
    def loop_method(cls) -> str:
        return "apply_arithmetic"

    @classmethod
    def compute_method(cls) -> str:
        return "_compute_arithmetic"

    @classmethod
    def probe_feature_set(cls) -> Any:
        return make_feature_set("v__add_constant", constant=1)

    @classmethod
    def name_probes(cls) -> list[tuple[str, str]]:
        return [("v__add_constant", "add")]

    @classmethod
    def option_probes(cls) -> list[tuple[str, Options, str]]:
        return [("my_result", Options(context={"arithmetic_op": "add", "in_features": "v", "constant": 1}), "add")]
