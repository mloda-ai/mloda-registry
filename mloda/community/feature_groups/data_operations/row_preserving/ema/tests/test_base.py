"""Tests for EmaFeatureGroup base class.

Covers the arity contract of the scalar ``order_by`` key: core unwraps a
one-element container when it reads a property value, so ``("timestamp",)`` is
valid caller syntax for one column and must dispatch to ``"timestamp"`` rather
than to the container's string form. Match tests use ``EmaFeatureGroup`` (the ema
convention, mirroring ``TestEmaMatchFeatureGroupCriteria`` in test_integration).
"""

from __future__ import annotations

from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from mloda.provider import FeatureChainParserMixin, FeatureSet
from mloda.user import Feature, Options

from mloda.community.feature_groups.data_operations.base import KeyColumnInputsMixin
from mloda.community.feature_groups.data_operations.row_preserving.ema.base import EmaFeatureGroup
from mloda.community.feature_groups.data_operations.row_preserving.ema.pandas_ema import PandasEma
from mloda.testing.data_creator.base import DataOperationsTestDataCreator
from mloda.testing.feature_groups.data_operations.helpers import (
    calculate_with,
    extract_column,
    feature_set_for,
    make_feature_set,
)
from mloda.testing.feature_groups.data_operations.match_validation import ScalarArityTestBase, TokenCase
from mloda.testing.feature_groups.data_operations.operation_group import ImplSpec, OperationGroupTestBase

FEATURE_NAME = "value_float__ema_2"


class TestOrderByArity(ScalarArityTestBase):
    """``order_by`` is a scalar key: one column, bare or in a single-element container.

    ema declares no operation config key (the span is part of the feature name), so it
    declares the arity base directly instead of ``MatchValidationTestBase``.
    """

    @classmethod
    def feature_group_class(cls) -> Any:
        return EmaFeatureGroup

    @classmethod
    def match_class(cls) -> Any:
        return EmaFeatureGroup

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
        return [EmaFeatureGroup._extract_order_by(Feature(FEATURE_NAME, options=options))]

    @classmethod
    def compute_values(cls, options: Options) -> list[Any] | None:
        # Compared as strings so that NaN, which never equals itself, stays comparable.
        df = pd.DataFrame(DataOperationsTestDataCreator.get_raw_data())
        result = calculate_with(PandasEma, df, feature_set_for(FEATURE_NAME, options))
        return [str(value) for value in extract_column(result, FEATURE_NAME)]


_ENGINE = "mloda_plugins.compute_framework.base_implementations"


class TestEmaOperationGroup(OperationGroupTestBase):
    @classmethod
    def group_class(cls) -> Any:
        return EmaFeatureGroup

    @classmethod
    def backend_package(cls) -> str:
        return "mloda.community.feature_groups.data_operations.row_preserving.ema"

    @classmethod
    def implementations(cls) -> list[ImplSpec]:
        return [
            ImplSpec(
                "pandas",
                "pandas_ema",
                "PandasEma",
                f"{_ENGINE}.pandas.dataframe",
                "PandasDataFrame",
                dependency="pandas",
            ),
            ImplSpec(
                "polars_lazy",
                "polars_lazy_ema",
                "PolarsLazyEma",
                f"{_ENGINE}.polars.lazy_dataframe",
                "PolarsLazyDataFrame",
                dependency="polars",
            ),
            ImplSpec(
                "python_dict",
                "python_dict_ema",
                "PythonDictEma",
                f"{_ENGINE}.python_dict.python_dict_framework",
                "PythonDictFramework",
            ),
        ]

    @classmethod
    def subtype_key(cls) -> None:
        """Ema has no subtype dimension."""
        return None

    @classmethod
    def universe(cls) -> frozenset[str]:
        return frozenset()

    @classmethod
    def loop_method(cls) -> str:
        return "ema"

    @classmethod
    def compute_method(cls) -> str:
        return "_compute_ema"

    @classmethod
    def probe_feature_set(cls) -> FeatureSet:
        return make_feature_set("value_float__ema_2", order_by="timestamp", partition_by=["region"])

    @classmethod
    def mixins(cls) -> tuple[type, ...]:
        return (KeyColumnInputsMixin, FeatureChainParserMixin)

    @classmethod
    def name_probes(cls) -> list[tuple[str, str]]:
        return []

    def test_name_only_match_is_declared(self) -> None:
        assert EmaFeatureGroup.NAME_ONLY_MATCH is True

    @classmethod
    def option_probes(cls) -> list[tuple[str, Options, str]]:
        return []

    @classmethod
    def unresolved_probes(cls) -> list[tuple[str, Options]]:
        return [("value_float__ema_2", Options())]
