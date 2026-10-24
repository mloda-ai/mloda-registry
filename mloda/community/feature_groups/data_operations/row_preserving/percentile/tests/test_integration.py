"""Integration tests for percentile through mloda's full pipeline.

PyArrow has no native grouped percentile, so percentile runs on pandas here: the group is
served by PandasPercentile. The tests verify that percentile operations work end-to-end through mloda's
runtime, including plugin discovery, feature resolution, and PluginCollector.
"""

from __future__ import annotations

from typing import Any

import pytest
from mloda.user import Feature, FeatureResolutionError, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

pd = pytest.importorskip("pandas")

from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame  # noqa: E402

from mloda.community.feature_groups.data_operations.row_preserving.percentile.base import (  # noqa: E402
    PercentileFeatureGroup,
)
from mloda.testing.data_creator.pandas import PandasDataOpsTestDataCreator  # noqa: E402
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator  # noqa: E402
from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase  # noqa: E402
from mloda.testing.feature_groups.data_operations.mixins.mask_integration import (  # noqa: E402
    MaskIntegrationTestMixin,
)


class TestPercentileIntegration(MaskIntegrationTestMixin, DataOpsIntegrationTestBase):
    @classmethod
    def feature_group_class(cls) -> type:
        return PercentileFeatureGroup

    @classmethod
    def data_creator_class(cls) -> type:
        return PandasDataOpsTestDataCreator

    @classmethod
    def compute_framework_class(cls) -> type:
        return PandasDataFrame

    @classmethod
    def no_implementation_framework(cls) -> type | None:
        return PyArrowTable

    @classmethod
    def primary_feature_name(cls) -> str:
        return "value_int__p50_percentile"

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"]}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        return [5.0, 5.0, 5.0, 5.0, 50.0, 50.0, 50.0, 50.0, 15.0, 15.0, 15.0, -10.0]

    @classmethod
    def secondary_feature_name(cls) -> str:
        return "value_int__p75_percentile"

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"]}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        return [12.5, 12.5, 12.5, 12.5, 55.0, 55.0, 55.0, 55.0, 27.5, 27.5, 27.5, -10.0]

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return ["value_int__p25_percentile", "value_int__p50_percentile", "value_int__p75_percentile"]

    @classmethod
    def invalid_feature_names(cls) -> list[str]:
        return ["value_int", "value_int__p50"]

    @classmethod
    def match_options(cls) -> Options:
        return Options(context={"partition_by": ["region"]})

    @classmethod
    def use_approx(cls) -> bool:
        return True

    # -- MaskIntegrationTestMixin configuration --------------------------------

    @classmethod
    def mask_integration_feature_name(cls) -> str:
        return "value_int__p50_percentile"

    @classmethod
    def mask_integration_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"]}

    @classmethod
    def mask_integration_expected(cls) -> list[Any]:
        # category='X': A=[10,0] p50=5.0, B=[60] p50=60.0, C=[15] p50=15.0, None=[-10] p50=-10.0
        return [5.0, 5.0, 5.0, 5.0, 60.0, 60.0, 60.0, 60.0, 15.0, 15.0, 15.0, -10.0]

    @classmethod
    def mask_integration_complex_expected(cls) -> list[Any]:
        # category='X' AND value_int>=10: A=[10] p50=10, B=[60] p50=60, C=[15] p50=15, None=[] p50=None
        return [10.0, 10.0, 10.0, 10.0, 60.0, 60.0, 60.0, 60.0, 15.0, 15.0, 15.0, None]

    @classmethod
    def mask_integration_unmasked_expected(cls) -> list[Any]:
        return [5.0, 5.0, 5.0, 5.0, 50.0, 50.0, 50.0, 50.0, 15.0, 15.0, 15.0, -10.0]

    @classmethod
    def mask_integration_use_approx(cls) -> bool:
        return True

    def test_own_contradictory_percentile_raises_name_agreement_error(self) -> None:
        """The error is escalated: a contained match error would surface as FeatureResolutionError."""
        feature = Feature(
            "value_int__p50_percentile", options=Options(context={"partition_by": ["region"], "percentile": 0.75})
        )

        with pytest.raises(
            ValueError, match=r"option 'percentile' is float 0\.75, but the feature name encodes '0\.5'"
        ) as exc:
            mloda.run_all(
                [feature],
                compute_frameworks=[self.compute_framework_class()],
                plugin_collector=self._plugin_collector(),
            )
        assert not isinstance(exc.value, FeatureResolutionError)


class TestIntegrationMultipleFeatures:
    def test_p50_and_p75_together(self) -> None:
        """Request both p50 and p75 in one pipeline run."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PandasDataOpsTestDataCreator, PercentileFeatureGroup}
        )

        f_p50 = Feature(
            "value_int__p50_percentile",
            options=Options(context={"partition_by": ["region"]}),
        )
        f_p75 = Feature(
            "value_int__p75_percentile",
            options=Options(context={"partition_by": ["region"]}),
        )

        results = mloda.run_all(
            [f_p50, f_p75],
            compute_frameworks=[PandasDataFrame],
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        p50_found = False
        p75_found = False
        for table in results:
            if not isinstance(table, pd.DataFrame):
                continue
            if "value_int__p50_percentile" in table.columns:
                p50_col = table["value_int__p50_percentile"].tolist()
                expected_p50 = [5.0, 5.0, 5.0, 5.0, 50.0, 50.0, 50.0, 50.0, 15.0, 15.0, 15.0, -10.0]
                assert p50_col == pytest.approx(expected_p50, rel=1e-3)
                p50_found = True
            if "value_int__p75_percentile" in table.columns:
                p75_col = table["value_int__p75_percentile"].tolist()
                expected_p75 = [12.5, 12.5, 12.5, 12.5, 55.0, 55.0, 55.0, 55.0, 27.5, 27.5, 27.5, -10.0]
                assert p75_col == pytest.approx(expected_p75, rel=1e-3)
                p75_found = True

        assert p50_found, "p50_percentile result not found in any result table"
        assert p75_found, "p75_percentile result not found in any result table"

    def test_different_percentiles(self) -> None:
        """Request p25 and p100 in one pipeline run."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PandasDataOpsTestDataCreator, PercentileFeatureGroup}
        )

        f_p25 = Feature(
            "value_int__p25_percentile",
            options=Options(context={"partition_by": ["region"]}),
        )
        f_p100 = Feature(
            "value_int__p100_percentile",
            options=Options(context={"partition_by": ["region"]}),
        )

        results = mloda.run_all(
            [f_p25, f_p100],
            compute_frameworks=[PandasDataFrame],
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        p25_found = False
        p100_found = False
        for table in results:
            if not isinstance(table, pd.DataFrame):
                continue
            if "value_int__p25_percentile" in table.columns:
                p25_col = table["value_int__p25_percentile"].tolist()
                expected_p25 = [-1.25, -1.25, -1.25, -1.25, 40.0, 40.0, 40.0, 40.0, 15.0, 15.0, 15.0, -10.0]
                assert p25_col == pytest.approx(expected_p25, rel=1e-3)
                p25_found = True
            if "value_int__p100_percentile" in table.columns:
                p100_col = table["value_int__p100_percentile"].tolist()
                expected_p100 = [20.0, 20.0, 20.0, 20.0, 60.0, 60.0, 60.0, 60.0, 40.0, 40.0, 40.0, -10.0]
                assert p100_col == pytest.approx(expected_p100, rel=1e-3)
                p100_found = True

        assert p25_found, "p25_percentile result not found in any result table"
        assert p100_found, "p100_percentile result not found in any result table"


class TestIntegrationSeveralFrameworks:
    """One PercentileFeatureGroup is served by one Implementation per framework, chosen per feature."""

    @staticmethod
    def _feature(name: str) -> Feature:
        return Feature(name, options=Options(context={"partition_by": ["region"]}))

    def test_several_frameworks_enabled_runs_p50(self) -> None:
        pytest.importorskip("polars")
        from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

        name = "value_int__p50_percentile"
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PandasDataOpsTestDataCreator, PercentileFeatureGroup}
        )

        results = mloda.run_all(
            [self._feature(name)],
            compute_frameworks=[PandasDataFrame, PolarsLazyDataFrame],
            plugin_collector=plugin_collector,
        )

        frames = [f for f in results if isinstance(f, pd.DataFrame) and name in f.columns]
        assert len(frames) == 1
        expected = [5.0, 5.0, 5.0, 5.0, 50.0, 50.0, 50.0, 50.0, 15.0, 15.0, 15.0, -10.0]
        assert frames[0][name].tolist() == pytest.approx(expected, rel=1e-3)

    def test_pyarrow_alone_is_a_resolution_error(self) -> None:
        """No percentile Implementation serves PyArrow, so the group is unresolvable there."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PercentileFeatureGroup}
        )

        with pytest.raises(FeatureResolutionError):
            mloda.run_all(
                [self._feature("value_int__p50_percentile")],
                compute_frameworks=[PyArrowTable],
                plugin_collector=plugin_collector,
            )
