"""Integration tests for rank through mloda's full pipeline.

PyArrow has no native ranking functions, so rank runs on pandas here: the group is served by
PandasRank. The tests verify that rank operations work end-to-end through mloda's runtime,
including plugin discovery, feature resolution, and PluginCollector.
"""

from __future__ import annotations

from typing import Any

import pytest
from mloda.user import Feature, FeatureResolutionError, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.rank.base import (
    RankFeatureGroup,
)
from mloda.testing.data_creator.base import column_selective
from mloda.testing.data_creator.pandas import PandasDataOpsTestDataCreator
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator


@pytest.fixture
def pd() -> Any:
    """pandas, or a skip of the requesting test when it is not installed."""
    return pytest.importorskip("pandas")


needs_pandas = pytest.mark.usefixtures("pd")


def _ranked_column(name: str, creator: type = PandasDataOpsTestDataCreator, order_by: str = "value_int") -> list[Any]:
    """Run one rank feature on pandas through ``run_all`` and return its column."""
    pd = pytest.importorskip("pandas")
    plugin_collector = PluginCollector.enabled_feature_groups({creator, RankFeatureGroup})
    feature = Feature(name, options=Options(context={"partition_by": ["region"], "order_by": order_by}))

    results = mloda.run_all([feature], compute_frameworks=[PandasDataFrame], plugin_collector=plugin_collector)

    result_frame = next(f for f in results if isinstance(f, pd.DataFrame) and name in f.columns)
    assert len(result_frame) == 12
    return list(result_frame[name].tolist())


class TestIntegrationBasic:
    """Test rank features through the full mloda pipeline."""

    def test_row_number_through_pipeline(self) -> None:
        """Run value_int__row_number_ranked through run_all."""
        assert _ranked_column("value_int__row_number_ranked") == [3, 1, 2, 4, 4, 2, 1, 3, 1, 2, 3, 1]

    def test_dense_rank_through_pipeline(self) -> None:
        """Run value_int__dense_rank_ranked through run_all."""
        assert _ranked_column("value_int__dense_rank_ranked") == [3, 1, 2, 4, 4, 2, 1, 3, 1, 1, 2, 1]

    def test_top_n_through_pipeline(self) -> None:
        """Run value_int__top_3_ranked through run_all."""
        expected = [True, False, True, True, False, True, True, True, True, True, True, True]
        assert _ranked_column("value_int__top_3_ranked") == expected

    def test_row_number_with_column_selective_source(self) -> None:
        creator = column_selective(PandasDataOpsTestDataCreator)
        result = _ranked_column("value_int__row_number_ranked", creator=creator, order_by="timestamp")
        assert result == [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 1]


class TestIntegrationPluginDiscovery:
    """Test plugin discovery for rank feature groups."""

    def test_feature_group_is_discoverable(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({PandasDataOpsTestDataCreator, RankFeatureGroup})
        assert plugin_collector.applicable_feature_group_class(RankFeatureGroup)

    @needs_pandas
    def test_disabled_feature_group_blocks_execution(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({PandasDataOpsTestDataCreator})

        feature = Feature(
            "value_int__row_number_ranked",
            options=Options(context={"partition_by": ["region"], "order_by": "value_int"}),
        )

        with pytest.raises(ValueError):
            mloda.run_all(
                [feature],
                compute_frameworks=[PandasDataFrame],
                plugin_collector=plugin_collector,
            )

    def test_match_feature_group_criteria(self) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        assert RankFeatureGroup.match_feature_group_criteria("value_int__row_number_ranked", options)
        assert RankFeatureGroup.match_feature_group_criteria("value_int__rank_ranked", options)

    def test_match_rejects_missing_order_by(self) -> None:
        options = Options(context={"partition_by": ["region"]})
        assert not RankFeatureGroup.match_feature_group_criteria("value_int__row_number_ranked", options)

    def test_match_rejects_missing_partition_by(self) -> None:
        options = Options(context={"order_by": "value_int"})
        assert not RankFeatureGroup.match_feature_group_criteria("value_int__row_number_ranked", options)


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


class TestIntegrationSeveralFrameworks:
    """One RankFeatureGroup is served by one Implementation per framework, chosen per feature."""

    @staticmethod
    def _collector() -> PluginCollector:
        return PluginCollector.enabled_feature_groups({PandasDataOpsTestDataCreator, RankFeatureGroup})

    @staticmethod
    def _feature(name: str) -> Feature:
        return Feature(name, options=Options(context={"partition_by": ["region"], "order_by": "value_int"}))

    @staticmethod
    def _rank_step(feature: Feature, frameworks: list[Any], collector: PluginCollector) -> Any:
        plan = mloda.explain([feature], compute_frameworks=frameworks, plugin_collector=collector)
        steps = [step for step in plan if step.step_kind == "compute" and step.feature_group is RankFeatureGroup]
        assert len(steps) == 1
        return steps[0]

    def test_several_frameworks_enabled_runs_row_number(self, pd: Any) -> None:
        pytest.importorskip("polars")
        from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

        name = "value_int__row_number_ranked"
        results = mloda.run_all(
            [self._feature(name)],
            compute_frameworks=[PandasDataFrame, PolarsLazyDataFrame],
            plugin_collector=self._collector(),
        )

        frames = [f for f in results if isinstance(f, pd.DataFrame) and name in f.columns]
        assert len(frames) == 1
        assert frames[0][name].tolist() == [3, 1, 2, 4, 4, 2, 1, 3, 1, 2, 3, 1]

    @needs_pandas
    def test_divergent_pandas_rank_is_served_by_polars_lazy(self) -> None:
        """Pandas declares rank divergent (None and NaN tie), so data held in pandas is ranked by polars-lazy.

        row_number is not divergent on pandas, so the same data stays on pandas without a framework transform.
        """
        pytest.importorskip("polars")
        from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

        from mloda.community.feature_groups.data_operations.row_preserving.rank.pandas_rank import PandasRank
        from mloda.community.feature_groups.data_operations.row_preserving.rank.polars_lazy_rank import PolarsLazyRank

        frameworks = [PandasDataFrame, PolarsLazyDataFrame]

        rank_step = self._rank_step(self._feature("value_int__rank_ranked"), frameworks, self._collector())
        assert rank_step.implementation is PolarsLazyRank
        assert rank_step.compute_framework is PolarsLazyDataFrame

        row_number_step = self._rank_step(self._feature("value_int__row_number_ranked"), frameworks, self._collector())
        assert row_number_step.implementation is PandasRank
        assert row_number_step.compute_framework is PandasDataFrame

    def test_pyarrow_alone_is_a_resolution_error(self) -> None:
        """No rank Implementation serves PyArrow, so the group is unresolvable there, not a NotImplementedError."""
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, RankFeatureGroup})

        with pytest.raises(FeatureResolutionError):
            mloda.run_all(
                [self._feature("value_int__row_number_ranked")],
                compute_frameworks=[PyArrowTable],
                plugin_collector=plugin_collector,
            )
