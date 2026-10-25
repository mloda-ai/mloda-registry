"""Integration tests for rank through mloda's full pipeline.

PyArrow has no native ranking functions, so rank runs on pandas here: the group is served by
PandasRank. The shared base and operation-group mixin cover the pipeline; the rank-specific pins follow.
"""

from __future__ import annotations

from typing import Any

import pytest
from mloda.user import Feature, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.rank.base import (
    RankFeatureGroup,
)
from mloda.community.feature_groups.data_operations.row_preserving.rank.manifest import (
    FEATURE_GROUPS as _LOADED_FEATURE_GROUPS,  # noqa: F401  loads the installed implementations
)
from mloda.testing.data_creator.pandas import PandasDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.helpers import extract_column
from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase
from mloda.testing.feature_groups.data_operations.mixins.operation_group_integration import (
    PANDAS_DATAFRAME,
    OperationGroupIntegrationTestMixin,
    RoutingCase,
)

POLARS_LAZY_DATAFRAME = "mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe:PolarsLazyDataFrame"


@pytest.fixture
def pd() -> Any:
    """pandas, or a skip of the requesting test when it is not installed."""
    return pytest.importorskip("pandas")


needs_pandas = pytest.mark.usefixtures("pd")


@needs_pandas
class TestRankIntegration(OperationGroupIntegrationTestMixin, DataOpsIntegrationTestBase):
    @classmethod
    def feature_group_class(cls) -> type:
        return RankFeatureGroup

    @classmethod
    def data_creator_class(cls) -> type:
        return PandasDataOpsTestDataCreator

    @classmethod
    def compute_framework_class(cls) -> type:
        return PandasDataFrame

    @classmethod
    def several_frameworks(cls) -> list[type[Any]]:
        pytest.importorskip("polars")
        from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

        return [PandasDataFrame, PolarsLazyDataFrame]

    @classmethod
    def routing_frameworks(cls) -> list[type[Any]]:
        pytest.importorskip("polars")
        from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

        return [PolarsLazyDataFrame]

    @classmethod
    def primary_feature_name(cls) -> str:
        return "value_int__row_number_ranked"

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"], "order_by": "timestamp"}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        return [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 1]

    @classmethod
    def secondary_feature_name(cls) -> str:
        return "value_int__dense_rank_ranked"

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"], "order_by": "value_int"}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        return [3, 1, 2, 4, 4, 2, 1, 3, 1, 1, 2, 1]

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return ["value_int__row_number_ranked", "value_int__rank_ranked"]

    @classmethod
    def invalid_feature_names(cls) -> list[str]:
        return ["value_int", "value_int__rank_window"]

    @classmethod
    def match_options(cls) -> Options:
        return Options(context={"partition_by": ["region"], "order_by": "value_int"})

    # -- OperationGroupIntegrationTestMixin configuration ----------------------

    @classmethod
    def divergent_routing_cases(cls) -> list[RoutingCase]:
        package = "mloda.community.feature_groups.data_operations.row_preserving.rank"
        options = {"partition_by": ["region"], "order_by": "value_int"}
        return [
            RoutingCase(
                "value_int__rank_ranked",
                options,
                f"{package}.polars_lazy_rank:PolarsLazyRank",
                POLARS_LAZY_DATAFRAME,
            ),
            RoutingCase(
                "value_int__row_number_ranked",
                options,
                f"{package}.pandas_rank:PandasRank",
                PANDAS_DATAFRAME,
            ),
        ]

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            pytest.param(
                "value_int__row_number_ranked", [3, 1, 2, 4, 4, 2, 1, 3, 1, 2, 3, 1], id="row_number_by_value_int"
            ),
            pytest.param(
                "value_int__top_3_ranked",
                [True, False, True, True, False, True, True, True, True, True, True, True],
                id="top_n",
            ),
        ],
    )
    def test_rank_types_through_pipeline(self, name: str, expected: list[Any]) -> None:
        result_table = self._run_single_feature(name, {"partition_by": ["region"], "order_by": "value_int"})
        self._assert_values_equal(extract_column(result_table, name), expected)


class TestIntegrationGuardRejectionReported:
    """A guard-rejected order_by must be named in the run_all resolution error."""

    @pytest.mark.parametrize(
        ("order_by", "match"),
        [
            pytest.param(123, r"option 'order_by' must be .*got int 123", id="config_path_mistyped"),
            pytest.param(["timestamp", "region"], r"option 'order_by' must be exactly one.*got list", id="arity"),
        ],
    )
    @needs_pandas
    def test_guard_rejected_order_by_reported_at_discovery(self, order_by: object, match: str) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({PandasDataOpsTestDataCreator, RankFeatureGroup})
        options = Options(
            context={
                "rank_type": "row_number",
                "in_features": "value_float",
                "partition_by": ["region"],
                "order_by": order_by,
            }
        )

        with pytest.raises(ValueError, match=match):
            mloda.run_all(
                [Feature("my_result", options=options)],
                compute_frameworks=[PandasDataFrame],
                plugin_collector=plugin_collector,
            )
