"""Tests for FfillFeatureGroup base class.

Covers the arity contract of the scalar ``order_by`` key: core unwraps a
one-element container when it reads a property value, so ``("timestamp",)`` is
valid caller syntax for one column and must dispatch to ``"timestamp"`` rather
than to the container's string form. Match tests use ``FfillFeatureGroup`` itself.
"""

from __future__ import annotations

from typing import Any

from mloda.provider import FeatureChainParserMixin, FeatureSet
from mloda.user import Feature, Options

from mloda.community.feature_groups.data_operations.base import KeyColumnInputsMixin, NameOnlyMatchMixin
from mloda.community.feature_groups.data_operations.row_preserving.ffill.base import FfillFeatureGroup
from mloda.community.feature_groups.data_operations.row_preserving.ffill.pyarrow_ffill import PyArrowFfill
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.helpers import (
    calculate_with,
    extract_column,
    feature_set_for,
    make_feature_set,
)
from mloda.testing.feature_groups.data_operations.match_validation import ScalarArityTestBase, TokenCase
from mloda.testing.feature_groups.data_operations.operation_group import ImplSpec, OperationGroupTestBase

FEATURE_NAME = "amount__ffill"


class TestOrderByArity(ScalarArityTestBase):
    """``order_by`` is a scalar key: one column, bare or in a single-element container.

    ffill is a single-op family with no operation config key at all, so it declares the
    arity base directly instead of ``MatchValidationTestBase``.
    """

    @classmethod
    def feature_group_class(cls) -> Any:
        return FfillFeatureGroup

    @classmethod
    def match_class(cls) -> Any:
        return FfillFeatureGroup

    @classmethod
    def match_feature_name(cls) -> str:
        return FEATURE_NAME

    @classmethod
    def base_context(cls) -> dict[str, Any]:
        return {"partition_by": ["region"]}

    @classmethod
    def token_cases(cls) -> list[TokenCase]:
        return [TokenCase("order_by", "timestamp", "region", required=True)]

    @classmethod
    def dispatch_values(cls, options: Options) -> list[Any]:
        return [FfillFeatureGroup._extract_order_by(Feature(FEATURE_NAME, options=options))]

    @classmethod
    def compute_values(cls, options: Options) -> list[Any] | None:
        result = calculate_with(
            PyArrowFfill, PyArrowDataOpsTestDataCreator.create(), feature_set_for(FEATURE_NAME, options)
        )
        return extract_column(result, FEATURE_NAME)


_ENGINE = "mloda_plugins.compute_framework.base_implementations"


class TestFfillOperationGroup(OperationGroupTestBase):
    @classmethod
    def group_class(cls) -> Any:
        return FfillFeatureGroup

    @classmethod
    def backend_package(cls) -> str:
        return "mloda.community.feature_groups.data_operations.row_preserving.ffill"

    @classmethod
    def implementations(cls) -> list[ImplSpec]:
        return [
            ImplSpec(
                "duckdb",
                "duckdb_ffill",
                "DuckdbFfill",
                f"{_ENGINE}.duckdb.duckdb_framework",
                "DuckDBFramework",
                dependency="duckdb",
            ),
            ImplSpec(
                "pandas",
                "pandas_ffill",
                "PandasFfill",
                f"{_ENGINE}.pandas.dataframe",
                "PandasDataFrame",
                dependency="pandas",
            ),
            ImplSpec(
                "polars_lazy",
                "polars_lazy_ffill",
                "PolarsLazyFfill",
                f"{_ENGINE}.polars.lazy_dataframe",
                "PolarsLazyDataFrame",
                dependency="polars",
            ),
            ImplSpec(
                "pyarrow",
                "pyarrow_ffill",
                "PyArrowFfill",
                f"{_ENGINE}.pyarrow.table",
                "PyArrowTable",
            ),
            ImplSpec(
                "python_dict",
                "python_dict_ffill",
                "PythonDictFfill",
                f"{_ENGINE}.python_dict.python_dict_framework",
                "PythonDictFramework",
            ),
            ImplSpec(
                "sqlite",
                "sqlite_ffill",
                "SqliteFfill",
                f"{_ENGINE}.sqlite.sqlite_framework",
                "SqliteFramework",
            ),
        ]

    @classmethod
    def subtype_key(cls) -> None:
        """Ffill has no subtype dimension."""
        return None

    @classmethod
    def universe(cls) -> frozenset[str]:
        return frozenset()

    @classmethod
    def loop_method(cls) -> str:
        return "ffill"

    @classmethod
    def compute_method(cls) -> str:
        return "_compute_ffill"

    @classmethod
    def probe_feature_set(cls) -> FeatureSet:
        return make_feature_set("value_float__ffill", order_by="timestamp", partition_by=["region"])

    @classmethod
    def mixins(cls) -> tuple[type, ...]:
        return (KeyColumnInputsMixin, NameOnlyMatchMixin, FeatureChainParserMixin)

    @classmethod
    def name_probes(cls) -> list[tuple[str, str]]:
        return []

    @classmethod
    def option_probes(cls) -> list[tuple[str, Options, str]]:
        return []

    @classmethod
    def unresolved_probes(cls) -> list[tuple[str, Options]]:
        return [("value_float__ffill", Options())]
