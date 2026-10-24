"""Integration tests for aggregation through mloda's full pipeline.

These tests verify that aggregation feature groups work end-to-end
through mloda's runtime, including plugin discovery, feature resolution,
and the PluginCollector mechanism.

Uses the shared DataOpsIntegrationTestBase from the testing library,
plus custom multi-feature tests for aggregation specifics.

Note: mloda's pipeline only returns requested feature columns, so
integration tests verify values as sets (not keyed by partition columns).
"""

from __future__ import annotations

import importlib
import re
from typing import Any

import pyarrow as pa
import pytest
from mloda.user import DataType, Feature, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.aggregation.base import (
    AggregationFeatureGroup,
)
from mloda.testing.data_creator.pandas import PandasDataOpsTestDataCreator
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.aggregation.aggregation import (
    EXPECTED_AVG_BY_REGION,
    EXPECTED_MAX_BY_REGION,
    EXPECTED_SUM_BY_REGION,
)
from mloda.testing.feature_groups.data_operations.integration import (
    DataOpsIntegrationTestBase,
    enabled_with_implementations,
)
from mloda.testing.feature_groups.data_operations.mixins.mask_integration import MaskIntegrationTestMixin
from mloda.testing.feature_groups.data_operations.mixins.operation_group_integration import (
    PANDAS_DATAFRAME,
    PYARROW_TABLE,
    OperationGroupIntegrationTestMixin,
    RoutingCase,
)


class TestAggregationIntegration(
    OperationGroupIntegrationTestMixin, MaskIntegrationTestMixin, DataOpsIntegrationTestBase
):
    """Standard integration tests inherited from the base class."""

    @classmethod
    def feature_group_class(cls) -> type:
        return AggregationFeatureGroup

    @classmethod
    def data_creator_class(cls) -> type:
        return PyArrowDataOpsTestDataCreator

    @classmethod
    def compute_framework_class(cls) -> type:
        return PyArrowTable

    @classmethod
    def primary_feature_name(cls) -> str:
        return "value_int__sum_agg"

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"]}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        return [25, 140, 70, -10]

    @classmethod
    def secondary_feature_name(cls) -> str:
        return "value_int__avg_agg"

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"]}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        return [6.25, 46.667, 23.333, -10.0]

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return ["value_int__sum_agg", "value_int__avg_agg", "value_int__count_agg"]

    @classmethod
    def invalid_feature_names(cls) -> list[str]:
        return ["value_int", "value_int__sum"]

    @classmethod
    def match_options(cls) -> Options:
        return Options(context={"partition_by": ["region"]})

    @classmethod
    def expected_row_count(cls) -> int:
        return 4

    @classmethod
    def compare_sorted(cls) -> bool:
        return True

    @classmethod
    def use_approx(cls) -> bool:
        return True

    # -- OperationGroupIntegrationTestMixin configuration ----------------------

    @classmethod
    def divergent_routing_cases(cls) -> list[RoutingCase]:
        package = "mloda.community.feature_groups.data_operations.aggregation"
        options = {"partition_by": ["region"]}
        return [
            RoutingCase(
                "value_int__sum_agg", options, f"{package}.pyarrow_aggregation:PyArrowAggregation", PYARROW_TABLE
            ),
            RoutingCase(
                "value_int__max_agg", options, f"{package}.pandas_aggregation:PandasAggregation", PANDAS_DATAFRAME
            ),
        ]

    @classmethod
    def unsupported_framework_case(cls) -> tuple[str, dict[str, Any]] | None:
        # PyArrow has no grouped median.
        return "value_int__median_agg", {"partition_by": ["region"]}

    # -- MaskIntegrationTestMixin configuration --------------------------------

    @classmethod
    def mask_integration_feature_name(cls) -> str:
        return "value_int__sum_agg"

    @classmethod
    def mask_integration_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"]}

    @classmethod
    def mask_integration_expected(cls) -> list[Any]:
        # category='X': sorted by value: [-10, 10, 15, 60]
        return sorted([-10, 10, 15, 60])

    @classmethod
    def mask_integration_complex_expected(cls) -> list[Any]:
        # category='X' AND value_int>=10: A=10, B=60, C=15, None=None -> sorted
        return sorted([10, 15, 60, None], key=lambda x: (x is None, x if x is not None else 0))

    @classmethod
    def mask_integration_unmasked_expected(cls) -> list[Any]:
        return sorted([25, 140, 70, -10])

    @classmethod
    def mask_integration_expected_row_count(cls) -> int:
        return 4

    @classmethod
    def mask_integration_compare_sorted(cls) -> bool:
        return True


class TestIntegrationAdditionalAggTypes:
    """Test additional aggregation types through the full mloda pipeline.

    Aggregation reduces rows. Several aggregations with the same ``partition_by``
    can share one run_all call (they are computed together into one result);
    aggregations with different ``partition_by`` need separate run_all calls
    (mloda cannot merge different row-count outputs without explicit Links).
    """

    def test_min_agg_through_pipeline(self) -> None:
        """Run value_int__min_agg through run_all and verify the result."""
        plugin_collector = enabled_with_implementations({PyArrowDataOpsTestDataCreator, AggregationFeatureGroup})

        feature = Feature(
            "value_int__min_agg",
            options=Options(context={"partition_by": ["region"]}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks=[PyArrowTable],
            plugin_collector=plugin_collector,
        )

        result_table = None
        for table in results:
            if isinstance(table, pa.Table) and "value_int__min_agg" in table.column_names:
                result_table = table
                break

        assert result_table is not None
        assert result_table.num_rows == 4
        min_col = sorted(result_table.column("value_int__min_agg").to_pylist())
        assert min_col == sorted([-5, 30, 15, -10])

    def test_max_agg_through_pipeline(self) -> None:
        """Run value_int__max_agg through run_all and verify the result."""
        plugin_collector = enabled_with_implementations({PyArrowDataOpsTestDataCreator, AggregationFeatureGroup})

        feature = Feature(
            "value_int__max_agg",
            options=Options(context={"partition_by": ["region"]}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks=[PyArrowTable],
            plugin_collector=plugin_collector,
        )

        result_table = None
        for table in results:
            if isinstance(table, pa.Table) and "value_int__max_agg" in table.column_names:
                result_table = table
                break

        assert result_table is not None
        assert result_table.num_rows == 4
        max_col = sorted(result_table.column("value_int__max_agg").to_pylist())
        assert max_col == sorted([20, 60, 40, -10])

    def test_std_agg_through_pipeline(self) -> None:
        """Run value_int__std_agg through run_all and verify the result."""
        plugin_collector = enabled_with_implementations({PyArrowDataOpsTestDataCreator, AggregationFeatureGroup})

        feature = Feature(
            "value_int__std_agg",
            options=Options(context={"partition_by": ["region"]}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks=[PyArrowTable],
            plugin_collector=plugin_collector,
        )

        result_table = None
        for table in results:
            if isinstance(table, pa.Table) and "value_int__std_agg" in table.column_names:
                result_table = table
                break

        assert result_table is not None
        assert result_table.num_rows == 4

    def test_nunique_agg_through_pipeline(self) -> None:
        """Run value_int__nunique_agg through run_all and verify the result."""
        plugin_collector = enabled_with_implementations({PyArrowDataOpsTestDataCreator, AggregationFeatureGroup})

        feature = Feature(
            "value_int__nunique_agg",
            options=Options(context={"partition_by": ["region"]}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks=[PyArrowTable],
            plugin_collector=plugin_collector,
        )

        result_table = None
        for table in results:
            if isinstance(table, pa.Table) and "value_int__nunique_agg" in table.column_names:
                result_table = table
                break

        assert result_table is not None
        assert result_table.num_rows == 4


class TestIntegrationMedianAggType:
    """Median through the pipeline: PyArrow has no grouped median, so another framework must serve it.

    PyArrowAggregation does not declare median in ``supported``, so a PyArrow-only run is rejected at
    match time. With Pandas enabled as well, the group is served by the Pandas implementation instead.
    """

    def test_median_agg_through_pipeline_on_a_supporting_framework(self) -> None:
        """Median runs end-to-end on Pandas when it is enabled next to PyArrow."""
        pytest.importorskip("pandas")
        import pandas as pd
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

        plugin_collector = enabled_with_implementations({PyArrowDataOpsTestDataCreator, AggregationFeatureGroup})

        feature = Feature(
            "value_int__median_agg",
            options=Options(context={"partition_by": ["region"]}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks=[PyArrowTable, PandasDataFrame],
            plugin_collector=plugin_collector,
        )

        result_frame = None
        for frame in results:
            if isinstance(frame, pd.DataFrame) and "value_int__median_agg" in frame.columns:
                result_frame = frame
                break

        assert result_frame is not None
        assert len(result_frame) == 4

        median_col = sorted(result_frame["value_int__median_agg"].tolist())
        # Regions: A=[-5,0,10,20] median=5.0, B=[30,50,60,None] median=50.0,
        #          C=[15,15,40] median=15.0, None=[-10] median=-10.0
        assert median_col == pytest.approx(sorted([-10.0, 5.0, 15.0, 50.0]))


class TestIntegrationSeveralFrameworks:
    """One AggregationFeatureGroup is served by one Implementation per framework, chosen per feature.

    Before the operation-group shape every framework was its own feature group, so enabling several
    frameworks failed with "Multiple feature groups".
    """

    @staticmethod
    def _collector() -> PluginCollector:
        return enabled_with_implementations({PyArrowDataOpsTestDataCreator, AggregationFeatureGroup})

    @staticmethod
    def _feature(**kwargs: Any) -> Feature:
        return Feature(
            "value_int__sum_agg",
            options=Options(context={"partition_by": ["region"]}),
            **kwargs,
        )

    @staticmethod
    def _aggregation_step(features: list[Feature], frameworks: list[Any], collector: PluginCollector) -> Any:
        plan = mloda.explain(features, compute_frameworks=frameworks, plugin_collector=collector)
        steps = [step for step in plan if step.step_kind == "compute" and step.feature_group is AggregationFeatureGroup]
        assert len(steps) == 1
        return steps[0]

    @pytest.mark.parametrize("pin", ["qualified", "class"], ids=["by_qualified_name", "by_class"])
    def test_implementation_pin_runs_on_pandas(self, pin: str) -> None:
        """Feature(..., implementation=...) overrides the ranking and runs the pinned implementation.

        Core ships a PandasAggregation too, so the bare name is ambiguous once core's module is loaded.
        """
        pytest.importorskip("pandas")
        import pandas as pd
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

        from mloda.community.feature_groups.data_operations.aggregation.pandas_aggregation import PandasAggregation

        importlib.import_module("mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas")
        qualified = "mloda.community.feature_groups.data_operations.aggregation.pandas_aggregation:PandasAggregation"
        feature = self._feature(implementation=qualified if pin == "qualified" else PandasAggregation)
        frameworks = [PyArrowTable, PandasDataFrame]

        step = self._aggregation_step([feature], frameworks, self._collector())
        assert step.implementation is PandasAggregation
        assert step.compute_framework is PandasDataFrame

        results = mloda.run_all([feature], compute_frameworks=frameworks, plugin_collector=self._collector())
        frames = [frame for frame in results if isinstance(frame, pd.DataFrame)]
        assert len(frames) == 1
        assert sorted(frames[0]["value_int__sum_agg"].tolist()) == sorted([25, 140, 70, -10])

    def test_bare_implementation_name_is_ambiguous_with_core(self) -> None:
        """Core's PandasAggregation shares the bare name, so pinning by it names several loaded classes."""
        pytest.importorskip("pandas")
        from mloda.provider import ImplementationPinError
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

        importlib.import_module("mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas")
        feature = self._feature(implementation="PandasAggregation")
        with pytest.raises(ImplementationPinError, match="names several loaded classes"):
            self._aggregation_step([feature], [PyArrowTable, PandasDataFrame], self._collector())


class TestIntegrationMultipleFeatures:
    """Test multiple aggregation features, in separate pipeline runs and in one run.

    Features that share partition_by are aggregated together from the original
    input in a single run_all call, giving one result table. Features may also
    be requested in separate run_all calls. These tests verify that different
    aggregation types produce consistent, correct results from the
    same source data either way.
    """

    def test_sum_and_avg_in_one_run(self) -> None:
        """Sum and avg of value_int by region in ONE run_all call share one result table."""
        plugin_collector = enabled_with_implementations({PyArrowDataOpsTestDataCreator, AggregationFeatureGroup})
        options = Options(context={"partition_by": ["region"]})
        results = mloda.run_all(
            [Feature("value_int__sum_agg", options=options), Feature("value_int__avg_agg", options=options)],
            compute_frameworks=[PyArrowTable],
            plugin_collector=plugin_collector,
        )

        tables = [
            t
            for t in results
            if isinstance(t, pa.Table) and {"value_int__sum_agg", "value_int__avg_agg"} <= set(t.column_names)
        ]
        assert len(tables) == 1
        table = tables[0]
        assert table.num_rows == 4
        assert sorted(table.column("value_int__sum_agg").to_pylist()) == sorted(EXPECTED_SUM_BY_REGION.values())
        assert sorted(table.column("value_int__avg_agg").to_pylist()) == pytest.approx(
            sorted(EXPECTED_AVG_BY_REGION.values()), rel=1e-3
        )

    def test_sum_and_avg_consistent(self) -> None:
        """Run sum and avg separately and verify both produce correct results."""
        plugin_collector = enabled_with_implementations({PyArrowDataOpsTestDataCreator, AggregationFeatureGroup})

        f_sum = Feature(
            "value_int__sum_agg",
            options=Options(context={"partition_by": ["region"]}),
        )
        sum_results = mloda.run_all(
            [f_sum],
            compute_frameworks=[PyArrowTable],
            plugin_collector=plugin_collector,
        )

        sum_table = None
        for table in sum_results:
            if isinstance(table, pa.Table) and "value_int__sum_agg" in table.column_names:
                sum_table = table
                break
        assert sum_table is not None
        sum_col = sorted(sum_table.column("value_int__sum_agg").to_pylist())
        assert sum_col == sorted([25, 140, 70, -10])

        f_avg = Feature(
            "value_int__avg_agg",
            options=Options(context={"partition_by": ["region"]}),
        )
        avg_results = mloda.run_all(
            [f_avg],
            compute_frameworks=[PyArrowTable],
            plugin_collector=plugin_collector,
        )

        avg_table = None
        for table in avg_results:
            if isinstance(table, pa.Table) and "value_int__avg_agg" in table.column_names:
                avg_table = table
                break
        assert avg_table is not None
        avg_col = sorted(avg_table.column("value_int__avg_agg").to_pylist())
        expected_avg = sorted([6.25, 46.667, 23.333, -10.0])
        assert avg_col == pytest.approx(expected_avg, rel=1e-3)

        assert sum_table.num_rows == avg_table.num_rows == 4

    def test_min_and_max_consistent(self) -> None:
        """Run min and max separately and verify both produce correct results."""
        plugin_collector = enabled_with_implementations({PyArrowDataOpsTestDataCreator, AggregationFeatureGroup})

        f_min = Feature(
            "value_int__min_agg",
            options=Options(context={"partition_by": ["region"]}),
        )
        min_results = mloda.run_all(
            [f_min],
            compute_frameworks=[PyArrowTable],
            plugin_collector=plugin_collector,
        )

        min_table = None
        for table in min_results:
            if isinstance(table, pa.Table) and "value_int__min_agg" in table.column_names:
                min_table = table
                break
        assert min_table is not None
        min_col = sorted(min_table.column("value_int__min_agg").to_pylist())
        assert min_col == sorted([-5, 30, 15, -10])

        f_max = Feature(
            "value_int__max_agg",
            options=Options(context={"partition_by": ["region"]}),
        )
        max_results = mloda.run_all(
            [f_max],
            compute_frameworks=[PyArrowTable],
            plugin_collector=plugin_collector,
        )

        max_table = None
        for table in max_results:
            if isinstance(table, pa.Table) and "value_int__max_agg" in table.column_names:
                max_table = table
                break
        assert max_table is not None
        max_col = sorted(max_table.column("value_int__max_agg").to_pylist())
        assert max_col == sorted([20, 60, 40, -10])

        assert min_table.num_rows == max_table.num_rows == 4

    @pytest.mark.parametrize(
        "frameworks, features",
        [
            pytest.param(
                [PandasDataFrame, PyArrowTable],
                [
                    Feature("value_int__median_agg", options=Options(context={"partition_by": ["region"]})),
                    Feature("value_int__max_agg", options=Options(context={"partition_by": ["region"]})),
                ],
                id="median_max_split_by_framework",
            ),
            pytest.param(
                [PandasDataFrame],
                [
                    Feature(
                        "value_int__sum_agg",
                        options=Options(context={"partition_by": ["region"]}),
                        data_type=DataType.INT64,
                    ),
                    Feature(
                        "value_int__max_agg",
                        options=Options(context={"partition_by": ["region"]}),
                        data_type=DataType.DOUBLE,
                    ),
                ],
                id="split_by_data_type",
            ),
        ],
    )
    def test_split_feature_sets_error_names_group_remedy(self, frameworks: list[Any], features: list[Feature]) -> None:
        """Features of one request split into several FeatureSets raise an error naming Options(group=...)."""
        plugin_collector = enabled_with_implementations({PandasDataOpsTestDataCreator, AggregationFeatureGroup})
        with pytest.raises(ValueError, match=re.escape("Options(group=")):
            mloda.run_all(features, compute_frameworks=frameworks, plugin_collector=plugin_collector)

    def test_split_feature_sets_with_distinct_groups_work(self) -> None:
        """Giving each feature its own Options(group=...) computes both aggregates correctly."""
        plugin_collector = enabled_with_implementations({PandasDataOpsTestDataCreator, AggregationFeatureGroup})
        features = [
            Feature(
                "value_int__median_agg",
                options=Options(group={"g": "median"}, context={"partition_by": ["region"]}),
            ),
            Feature(
                "value_int__max_agg",
                options=Options(group={"g": "max"}, context={"partition_by": ["region"]}),
            ),
        ]
        results = mloda.run_all(
            features,
            compute_frameworks=[PandasDataFrame, PyArrowTable],
            plugin_collector=plugin_collector,
        )

        found: dict[str, list[Any]] = {}
        for t in results:
            frame = t.to_pandas() if isinstance(t, pa.Table) else t
            for name in ("value_int__median_agg", "value_int__max_agg"):
                if name in frame.columns:
                    found[name] = sorted(frame[name].tolist())
        assert found["value_int__median_agg"] == sorted([5.0, 50.0, 15.0, -10.0])
        assert found["value_int__max_agg"] == sorted(EXPECTED_MAX_BY_REGION.values())
