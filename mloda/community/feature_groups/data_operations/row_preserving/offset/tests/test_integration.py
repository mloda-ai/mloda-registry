"""Integration tests for offset through mloda's full pipeline.

PyArrow has no native window functions, so offset runs on pandas here: the group is served by
PandasOffset. The shared base and operation-group mixin cover plugin discovery, feature resolution and
PluginCollector.
"""

from __future__ import annotations

from typing import Any

import pytest
from mloda.user import Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.offset.base import (
    OffsetFeatureGroup,
)
from mloda.community.feature_groups.data_operations.row_preserving.offset.manifest import (
    FEATURE_GROUPS as _LOADED_FEATURE_GROUPS,  # noqa: F401  loads the installed implementations
)
from mloda.testing.data_creator.pandas import PandasDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase
from mloda.testing.feature_groups.data_operations.mixins.operation_group_integration import (
    OperationGroupIntegrationTestMixin,
)


@pytest.fixture
def pd() -> Any:
    """pandas, or a skip of the requesting test when it is not installed."""
    return pytest.importorskip("pandas")


needs_pandas = pytest.mark.usefixtures("pd")


@needs_pandas
class TestOffsetIntegration(OperationGroupIntegrationTestMixin, DataOpsIntegrationTestBase):
    @classmethod
    def feature_group_class(cls) -> type:
        return OffsetFeatureGroup

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
    def nan_is_null(cls) -> bool:
        return True

    @classmethod
    def primary_feature_name(cls) -> str:
        return "value_int__lag_1_offset"

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"], "order_by": "timestamp"}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        return [None, 10, -5, 0, None, None, 50, 30, None, 15, 15, None]

    @classmethod
    def secondary_feature_name(cls) -> str:
        return "value_int__first_value_offset"

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"], "order_by": "value_int"}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        # Smallest value_int per region (nulls last): A -5, B 30, C 15, no region -10.
        return [-5, -5, -5, -5, 30, 30, 30, 30, 15, 15, 15, -10]

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return ["value_int__lag_1_offset", "value_int__first_value_offset"]

    @classmethod
    def invalid_feature_names(cls) -> list[str]:
        return ["value_int", "value_int__lag_1_ranked"]

    @classmethod
    def match_options(cls) -> Options:
        return Options(context={"partition_by": ["region"], "order_by": "value_int"})
