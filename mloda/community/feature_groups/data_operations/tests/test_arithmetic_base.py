"""Structural guard that both arithmetic families share one base class.

The point- and scalar-arithmetic feature groups both subclass a single base
(``ArithmeticFeatureGroupBase`` in ``data_operations/row_preserving/arithmetic/base.py``) that
holds the common arithmetic-op extraction and numeric-source skeleton.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from typing import Any

import pytest
from mloda.provider import FeatureGroup, Implementation, OperationFeatureGroup
from mloda.user import Options

from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.base import (
    ARITHMETIC_OP_NAMES,
    ArithmeticFeatureGroupBase,
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


class TestArithmeticFeatureGroupBase:
    """Both arithmetic families derive from the shared ``ArithmeticFeatureGroupBase``."""

    def test_point_arithmetic_subclasses_base(self) -> None:
        """``PointArithmeticFeatureGroup`` inherits from the shared base."""
        assert issubclass(PointArithmeticFeatureGroup, ArithmeticFeatureGroupBase)

    def test_scalar_arithmetic_subclasses_base(self) -> None:
        """``ScalarArithmeticFeatureGroup`` inherits from the shared base."""
        assert issubclass(ScalarArithmeticFeatureGroup, ArithmeticFeatureGroupBase)


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
# (family, group, name feature, config options) per family
_FAMILIES: dict[str, tuple[Any, str, dict[str, Any]]] = {
    "point": (PointArithmeticFeatureGroup, "a&b__add_point", {"arithmetic_op": "add", "in_features": ["a", "b"]}),
    "scalar": (ScalarArithmeticFeatureGroup, "v__add_constant", {"arithmetic_op": "add", "in_features": "v"}),
}
_IMPLEMENTATIONS = [
    pytest.param(family, backend, id=f"{family}-{backend}") for family in _FAMILIES for backend in _FRAMEWORKS
]
_GROUPS = [pytest.param(family, id=family) for family in _FAMILIES]


def _context(family: str, config: dict[str, Any]) -> dict[str, Any]:
    """The family's config options; the scalar family also needs a constant."""
    return {**config, "constant": 1} if family == "scalar" else dict(config)


def _implementation(family: str, backend: str) -> Any:
    """The named arithmetic Implementation; skips when its optional backend is not installed."""
    name = f"{_CLASS_NAMES[backend]}{family.capitalize()}Arithmetic"
    module = f"{_FAMILY_PACKAGE}.{family}_arithmetic.{backend}_{family}_arithmetic"
    return getattr(pytest.importorskip(module), name)


class TestOperationGroupShape:
    """Each arithmetic family is one OperationFeatureGroup served by one Implementation per framework."""

    @pytest.mark.parametrize("family", _GROUPS)
    def test_group_is_a_concrete_operation_group(self, family: str) -> None:
        group = _FAMILIES[family][0]
        assert issubclass(group, OperationFeatureGroup)
        assert not inspect.isabstract(group)

    @pytest.mark.parametrize("family", _GROUPS)
    def test_group_declares_the_arithmetic_op_subtype_key(self, family: str) -> None:
        declaration = _FAMILIES[family][0].SUBTYPES
        assert declaration is not None
        assert declaration.key == "arithmetic_op"

    @pytest.mark.parametrize("family", _GROUPS)
    def test_group_owns_the_loop_and_no_per_framework_compute(self, family: str) -> None:
        group = _FAMILIES[family][0]
        assert callable(group.apply_arithmetic)
        assert not hasattr(group, "_compute_arithmetic")

    @pytest.mark.parametrize(("family", "backend"), _IMPLEMENTATIONS)
    def test_implementation_names_its_pair(self, family: str, backend: str) -> None:
        impl = _implementation(family, backend)
        framework_module, framework_name = _FRAMEWORKS[backend]
        framework = getattr(pytest.importorskip(framework_module), framework_name)
        assert issubclass(impl, Implementation)
        assert not issubclass(impl, FeatureGroup)
        assert impl.feature_group is _FAMILIES[family][0]
        assert impl.compute_framework is framework

    @pytest.mark.parametrize(("family", "backend"), _IMPLEMENTATIONS)
    def test_implementation_keeps_its_compute_arithmetic(self, family: str, backend: str) -> None:
        assert callable(_implementation(family, backend)._compute_arithmetic)


class TestSupportedAndDivergent:
    """Capability and divergence are declared on the shipped Implementations."""

    @pytest.mark.parametrize(("family", "backend"), _IMPLEMENTATIONS)
    def test_supported_is_unrestricted(self, family: str, backend: str) -> None:
        assert _implementation(family, backend).supported is None

    def test_sqlite_point_divide_is_the_only_divergence(self) -> None:
        for family, backend in ((f, b) for f in _FAMILIES for b in _FRAMEWORKS):
            divergent = set(_implementation(family, backend).divergent)
            assert divergent == ({"divide"} if (family, backend) == ("point", "sqlite") else set())

    @pytest.mark.parametrize(("family", "backend"), _IMPLEMENTATIONS)
    def test_divergent_is_inside_the_universe_with_a_reason(self, family: str, backend: str) -> None:
        divergent = _implementation(family, backend).divergent
        assert set(divergent) <= _FAMILIES[family][0].subtype_universe()
        assert all(isinstance(reason, str) and reason for reason in divergent.values())


class TestSubtypeResolution:
    """supported and divergent apply only if the group resolves the arithmetic op; a miss is silent None."""

    @pytest.mark.parametrize("family", _GROUPS)
    def test_resolves_from_the_feature_name(self, family: str) -> None:
        group, name, _ = _FAMILIES[family]
        assert group.resolve_subtype(name, Options()) == "add"

    @pytest.mark.parametrize("family", _GROUPS)
    def test_resolves_from_the_config_option(self, family: str) -> None:
        group, _, config = _FAMILIES[family]
        context = _context(family, config)
        assert group.resolve_subtype("my_result", Options(context=context)) == "add"

    @pytest.mark.parametrize("family", _GROUPS)
    def test_group_matches_name_and_config_forms(self, family: str) -> None:
        group, name, config = _FAMILIES[family]
        context = _context(family, config)
        assert group.match_feature_group_criteria(name, Options(), None) is True
        assert group.match_feature_group_criteria("my_result", Options(context=context), None) is True


class TestManifest:
    """The entry-point manifest lists each group and its Implementations."""

    @pytest.mark.parametrize("family", _GROUPS)
    def test_manifest_lists_the_group_and_every_installed_implementation(self, family: str) -> None:
        manifest = importlib.import_module(f"{_FAMILY_PACKAGE}.{family}_arithmetic.manifest")
        group = _FAMILIES[family][0]
        assert group in manifest.FEATURE_GROUPS
        listed = {cls.__name__ for cls in manifest.FEATURE_GROUPS if issubclass(cls, Implementation)}
        expected = {
            f"{_CLASS_NAMES[backend]}{family.capitalize()}Arithmetic"
            for backend in _FRAMEWORKS
            if backend not in _OPTIONAL_DEPENDENCIES or importlib.util.find_spec(_OPTIONAL_DEPENDENCIES[backend])
        }
        assert listed == expected
        implementations: list[Any] = [cls for cls in manifest.FEATURE_GROUPS if cls is not group]
        assert all(cls.feature_group is group for cls in implementations)
