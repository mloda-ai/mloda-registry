"""Integration tests for string operations through mloda's full pipeline."""

from __future__ import annotations

from typing import Any, Optional, Set, Type, Union

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import ComputeFramework, FeatureGroup, FeatureSet
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.string.pyarrow_string import (
    PyArrowStringOps,
)


class StringTestDataCreator(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"name", "value"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table(
            {
                "name": ["Alice", "Bob", "Charlie", None, "Eve"],
                "value": [1, 2, 3, 4, 5],
            }
        )

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}


class TestStringIntegration:
    def test_upper_through_pipeline(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({StringTestDataCreator, PyArrowStringOps})

        feature = Feature("name__upper", options=Options())

        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        found = False
        for table in results:
            if isinstance(table, pa.Table) and "name__upper" in table.column_names:
                result_col = table.column("name__upper").to_pylist()
                assert result_col == ["ALICE", "BOB", "CHARLIE", None, "EVE"]
                found = True

        assert found, "name__upper result not found in any result table"

    def test_length_through_pipeline(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({StringTestDataCreator, PyArrowStringOps})

        feature = Feature("name__length", options=Options())

        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        found = False
        for table in results:
            if isinstance(table, pa.Table) and "name__length" in table.column_names:
                result_col = table.column("name__length").to_pylist()
                assert result_col == [5, 3, 7, None, 3]
                found = True

        assert found, "name__length result not found in any result table"

    def test_multiple_string_features(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({StringTestDataCreator, PyArrowStringOps})

        f_upper = Feature("name__upper", options=Options())
        f_lower = Feature("name__lower", options=Options())

        results = mloda.run_all(
            [f_upper, f_lower],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        upper_found = False
        lower_found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "name__upper" in table.column_names:
                upper_col = table.column("name__upper").to_pylist()
                assert upper_col == ["ALICE", "BOB", "CHARLIE", None, "EVE"]
                upper_found = True
            if "name__lower" in table.column_names:
                lower_col = table.column("name__lower").to_pylist()
                assert lower_col == ["alice", "bob", "charlie", None, "eve"]
                lower_found = True

        assert upper_found, "name__upper result not found in any result table"
        assert lower_found, "name__lower result not found in any result table"

    def test_disabled_feature_group_blocks_execution(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({StringTestDataCreator})

        feature = Feature("name__upper", options=Options())

        with pytest.raises(ValueError):
            mloda.run_all(
                [feature],
                compute_frameworks={PyArrowTable},
                plugin_collector=plugin_collector,
            )
