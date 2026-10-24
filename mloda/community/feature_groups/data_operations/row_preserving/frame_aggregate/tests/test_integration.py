"""Integration tests for frame aggregate feature group.

Uses the shared DataOpsIntegrationTestBase from the testing library.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest
from mloda.user import Feature, FeatureResolutionError, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
    FrameAggregateFeatureGroup,
)
from mloda.testing.data_creator.pandas import PandasDataOpsTestDataCreator
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.integration import (
    DataOpsIntegrationTestBase,
    enabled_with_implementations,
)
from mloda.testing.feature_groups.data_operations.mixins.mask_integration import MaskIntegrationTestMixin


class TestFrameAggregateIntegration(MaskIntegrationTestMixin, DataOpsIntegrationTestBase):
    """Standard integration tests inherited from the base class.

    Uses the unified DataOpsIntegrationTestBase framework with real expected
    values so base-class test methods (primary/secondary pipeline, discovery,
    pattern matching) run without overrides.
    """

    @classmethod
    def feature_group_class(cls) -> type:
        return FrameAggregateFeatureGroup

    @classmethod
    def data_creator_class(cls) -> type:
        return PyArrowDataOpsTestDataCreator

    @classmethod
    def compute_framework_class(cls) -> type:
        return PyArrowTable

    @classmethod
    def pipeline_compute_frameworks(cls) -> list[type]:
        # No PyArrow Implementation exists: PyArrow data is served by the PythonDict frame Implementation.
        return [PyArrowTable, PythonDictFramework]

    @classmethod
    def primary_feature_name(cls) -> str:
        return "value_int__sum_rolling_3"

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"], "order_by": "value_int"}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        # Rolling sum (window 3) on value_int, partitioned by region, ordered by value_int.
        # Region A sorted: -5, 0, 10, 20
        # Region B sorted: 30, 50, 60, None
        # Region C sorted: 15, 15, 40
        # Region None: -10
        return [5, -5, -5, 30, 110, 80, 30, 140, 15, 30, 70, -10]

    @classmethod
    def secondary_feature_name(cls) -> str:
        return "value_int__cumsum"

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"], "order_by": "value_int"}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        # Cumulative sum on value_int, partitioned by region, ordered by value_int.
        return [5, -5, -5, 25, 140, 80, 30, 140, 15, 30, 70, -10]

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return [
            "value_int__sum_rolling_3",
            "value_int__avg_rolling_5",
            "value_int__cumsum",
            "value_int__cummax",
            "value_int__cumavg",
            "value_int__expanding_avg",
        ]

    @classmethod
    def invalid_feature_names(cls) -> list[str]:
        return [
            "value_int__sum_window",
            "value_int",
            "plain_feature",
        ]

    @classmethod
    def match_options(cls) -> Options:
        return Options(context={"partition_by": ["region"], "order_by": "value_int"})

    @classmethod
    def expected_row_count(cls) -> int:
        return 12

    # -- MaskIntegrationTestMixin configuration --------------------------------

    @classmethod
    def mask_integration_feature_name(cls) -> str:
        return "value_int__cumsum"

    @classmethod
    def mask_integration_options(cls) -> dict[str, Any]:
        # order_by differs from the masked source (value_int): masking an order key is not a frame contract.
        return {"partition_by": ["region"], "order_by": "timestamp"}

    @classmethod
    def mask_integration_expected(cls) -> list[Any]:
        # category='X' cumsum through pipeline; order_by is unmasked, so a masked row
        # is null until an earlier unmasked row enters its cumulative window
        return [10, None, 0, 10, 60, None, None, 60, None, 15, 15, -10]

    @classmethod
    def mask_integration_complex_expected(cls) -> list[Any]:
        # category='X' AND value_int>=10 through pipeline; order_by is unmasked
        return [10, None, None, 10, 60, None, None, 60, None, 15, 15, None]

    @classmethod
    def mask_integration_unmasked_expected(cls) -> list[Any]:
        return [10, 5, 5, 25, None, 50, 80, 140, 15, 30, 70, -10]

    def test_own_contradictory_frame_size_raises_name_agreement_error(self) -> None:
        """The error is escalated: a contained match error would surface as FeatureResolutionError."""
        feature = Feature(
            "value_int__sum_rolling_3",
            options=Options(context={"partition_by": ["region"], "order_by": "value_int", "frame_size": 5}),
        )

        with pytest.raises(ValueError, match=r"option 'frame_size' is int 5, but the feature name encodes '3'") as exc:
            mloda.run_all(
                [feature],
                compute_frameworks=self.pipeline_compute_frameworks(),
                plugin_collector=self._plugin_collector(),
                output_framework=self.compute_framework_class(),
            )
        assert not isinstance(exc.value, FeatureResolutionError)


class TestFrameAggregateMultiFeature:
    """Test multiple frame aggregate features in a single pipeline run."""

    def test_rolling_and_cumulative_together(self) -> None:
        plugin_collector = enabled_with_implementations({PyArrowDataOpsTestDataCreator, FrameAggregateFeatureGroup})

        features: list[Feature | str] = [
            Feature("value_int__sum_rolling_3", Options(context={"partition_by": ["region"], "order_by": "value_int"})),
            Feature("value_int__cumsum", Options(context={"partition_by": ["region"], "order_by": "value_int"})),
        ]

        results = mloda.run_all(
            features,
            compute_frameworks=[PyArrowTable, PythonDictFramework],
            plugin_collector=plugin_collector,
            output_framework=PyArrowTable,
        )

        assert len(results) >= 1

        rolling_found = False
        cumsum_found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "value_int__sum_rolling_3" in table.column_names:
                col = table.column("value_int__sum_rolling_3").to_pylist()
                assert col == [5, -5, -5, 30, 110, 80, 30, 140, 15, 30, 70, -10]
                rolling_found = True
            if "value_int__cumsum" in table.column_names:
                col = table.column("value_int__cumsum").to_pylist()
                assert col == [5, -5, -5, 25, 140, 80, 30, 140, 15, 30, 70, -10]
                cumsum_found = True

        assert rolling_found, "sum_rolling_3 result not found in any result table"
        assert cumsum_found, "cumsum result not found in any result table"

    def test_expanding_and_rolling_different_aggs(self) -> None:
        """Request expanding avg and rolling min in one pipeline run."""
        plugin_collector = enabled_with_implementations({PyArrowDataOpsTestDataCreator, FrameAggregateFeatureGroup})

        features: list[Feature | str] = [
            Feature(
                "value_int__expanding_avg",
                Options(context={"partition_by": ["region"], "order_by": "value_int"}),
            ),
            Feature(
                "value_int__min_rolling_2",
                Options(context={"partition_by": ["region"], "order_by": "value_int"}),
            ),
        ]

        results = mloda.run_all(
            features,
            compute_frameworks=[PyArrowTable, PythonDictFramework],
            plugin_collector=plugin_collector,
            output_framework=PyArrowTable,
        )

        assert len(results) >= 1

        # Expected expanding avg: see FrameAggregateTestBase EXPECTED_EXPANDING_AVG
        expected_expanding_avg = [
            5.0 / 3.0,
            -5.0,
            -2.5,
            6.25,
            140.0 / 3.0,
            40.0,
            30.0,
            140.0 / 3.0,
            15.0,
            15.0,
            70.0 / 3.0,
            -10.0,
        ]
        # Expected rolling min (window 2) on value_int, partitioned by region, ordered by value_int.
        expected_rolling_min_2 = [0, -5, -5, 10, 60, 30, 30, 50, 15, 15, 15, -10]

        expanding_found = False
        rolling_min_found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "value_int__expanding_avg" in table.column_names:
                col = table.column("value_int__expanding_avg").to_pylist()
                for i, (actual, expected) in enumerate(zip(col, expected_expanding_avg)):
                    assert actual == pytest.approx(expected, rel=1e-3), f"expanding_avg row {i}: {actual} != {expected}"
                expanding_found = True
            if "value_int__min_rolling_2" in table.column_names:
                col = table.column("value_int__min_rolling_2").to_pylist()
                assert col == expected_rolling_min_2
                rolling_min_found = True

        assert expanding_found, "expanding_avg result not found in any result table"
        assert rolling_min_found, "min_rolling_2 result not found in any result table"


class TestFrameAggregateFrameworkServing:
    """Frame has no PyArrow Implementation: frameworks decide which Implementation can serve a feature."""

    _OPTIONS = {"partition_by": ["region"], "order_by": "value_int"}

    def _collector(self) -> PluginCollector:
        return enabled_with_implementations({PyArrowDataOpsTestDataCreator, FrameAggregateFeatureGroup})

    def test_pyarrow_alone_has_no_implementation(self) -> None:
        feature = Feature("value_int__sum_rolling_3", options=Options(context=self._OPTIONS))
        with pytest.raises(FeatureResolutionError):
            mloda.run_all([feature], compute_frameworks=[PyArrowTable], plugin_collector=self._collector())

    def test_several_frameworks_enabled_runs(self) -> None:
        pytest.importorskip("pandas")
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

        feature = Feature("value_int__sum_rolling_3", options=Options(context=self._OPTIONS))
        results = mloda.run_all(
            [feature],
            compute_frameworks=[PyArrowTable, PandasDataFrame, PythonDictFramework],
            plugin_collector=self._collector(),
            output_framework=PyArrowTable,
        )
        tables = [t for t in results if isinstance(t, pa.Table) and "value_int__sum_rolling_3" in t.column_names]
        assert len(tables) == 1
        assert tables[0].column("value_int__sum_rolling_3").to_pylist() == [
            5,
            -5,
            -5,
            30,
            110,
            80,
            30,
            140,
            15,
            30,
            70,
            -10,
        ]


class TestFrameAggregateResolveTimeRejection:
    """A framework whose Implementation does not accept the feature is a resolution error, not a compute failure.

    Each case pairs the rejected feature with an accepted control on the same frameworks, so the
    error cannot come from the framework combination alone.
    """

    _OPTIONS = {"partition_by": ["region"], "order_by": "timestamp"}

    def _plan(self, name: str, creator: type, frameworks: list[type]) -> Any:
        collector = enabled_with_implementations({creator, FrameAggregateFeatureGroup})
        feature = Feature(name, options=Options(context=self._OPTIONS))
        return mloda.explain([feature], compute_frameworks=frameworks, plugin_collector=collector)

    def _assert_rejected_with_control(
        self, rejected: str, accepted: str, creator: type, frameworks: list[type]
    ) -> None:
        steps = [s for s in self._plan(accepted, creator, frameworks) if s.step_kind == "compute"]
        assert any(s.feature_group is FrameAggregateFeatureGroup for s in steps)
        with pytest.raises(FeatureResolutionError):
            self._plan(rejected, creator, frameworks)

    def test_pandas_month_window_is_rejected(self) -> None:
        pytest.importorskip("pandas")
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

        self._assert_rejected_with_control(
            "value_int__sum_1_month_window",
            "value_int__sum_3_day_window",
            PandasDataOpsTestDataCreator,
            [PandasDataFrame],
        )

    def test_polars_lazy_cumstd_is_rejected(self) -> None:
        pytest.importorskip("polars")
        from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

        self._assert_rejected_with_control(
            "value_int__cumstd",
            "value_int__cumsum",
            PyArrowDataOpsTestDataCreator,
            [PyArrowTable, PolarsLazyDataFrame],
        )
