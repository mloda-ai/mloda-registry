"""Integration tests for datetime extraction through mloda's full pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Set, Type, Union

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import ComputeFramework, FeatureGroup, FeatureSet
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.datetime.pyarrow_datetime import (
    PyArrowDateTimeExtraction,
)


class DateTimeTestDataCreator(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"timestamp", "value"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table(
            {
                "timestamp": [
                    datetime(2024, 1, 15, 10, 30, 45),
                    datetime(2024, 6, 22, 14, 0, 0),
                    datetime(2024, 12, 25, 0, 0, 0),
                    datetime(2024, 3, 9, 8, 15, 30),
                    datetime(2024, 7, 13, 18, 45, 59),
                ],
                "value": [1, 2, 3, 4, 5],
            }
        )

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}


class TestDateTimeIntegration:
    def test_year_extraction_through_pipeline(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({DateTimeTestDataCreator, PyArrowDateTimeExtraction})

        feature = Feature("timestamp__year", options=Options())

        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        found = False
        for table in results:
            if isinstance(table, pa.Table) and "timestamp__year" in table.column_names:
                result_col = table.column("timestamp__year").to_pylist()
                assert result_col == [2024, 2024, 2024, 2024, 2024]
                found = True

        assert found, "timestamp__year result not found in any result table"

    def test_is_weekend_through_pipeline(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({DateTimeTestDataCreator, PyArrowDateTimeExtraction})

        feature = Feature("timestamp__is_weekend", options=Options())

        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        found = False
        for table in results:
            if isinstance(table, pa.Table) and "timestamp__is_weekend" in table.column_names:
                result_col = table.column("timestamp__is_weekend").to_pylist()
                assert result_col == [0, 1, 0, 1, 1]
                found = True

        assert found, "timestamp__is_weekend result not found in any result table"

    def test_multiple_datetime_features(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({DateTimeTestDataCreator, PyArrowDateTimeExtraction})

        f_year = Feature("timestamp__year", options=Options())
        f_month = Feature("timestamp__month", options=Options())

        results = mloda.run_all(
            [f_year, f_month],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        year_found = False
        month_found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "timestamp__year" in table.column_names:
                year_col = table.column("timestamp__year").to_pylist()
                assert year_col == [2024, 2024, 2024, 2024, 2024]
                year_found = True
            if "timestamp__month" in table.column_names:
                month_col = table.column("timestamp__month").to_pylist()
                assert month_col == [1, 6, 12, 3, 7]
                month_found = True

        assert year_found, "timestamp__year result not found in any result table"
        assert month_found, "timestamp__month result not found in any result table"

    def test_disabled_feature_group_blocks_execution(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({DateTimeTestDataCreator})

        feature = Feature("timestamp__year", options=Options())

        with pytest.raises(ValueError):
            mloda.run_all(
                [feature],
                compute_frameworks={PyArrowTable},
                plugin_collector=plugin_collector,
            )
