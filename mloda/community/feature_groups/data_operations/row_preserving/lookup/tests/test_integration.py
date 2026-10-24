"""Integration tests for lookup through mloda's full pipeline (PyArrow).

Expected columns are pinned as literals on the canonical dataset, whose ``region``
column is A x4, B x4, C x3, None x1.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest
from mloda.user import Feature, FeatureResolutionError, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.lookup.base import (
    LookupFeatureGroup,
    LookupUnmatchedError,
)
from mloda.community.feature_groups.data_operations.row_preserving.lookup.pyarrow_lookup import PyArrowLookup
from mloda.testing.data_creator.pandas import PandasDataOpsTestDataCreator
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.helpers import assert_values_with_nulls
from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase

_MAPPING: dict[str, str] = {"A": "Alpha", "B": "Beta"}

_REGION_NULL: list[Any] = ["Alpha"] * 4 + ["Beta"] * 4 + [None] * 4
_REGION_KEEP: list[Any] = ["Alpha"] * 4 + ["Beta"] * 4 + ["C"] * 3 + [None]


class _RegionHookLookup(LookupFeatureGroup):
    """Hook subclass: matches ``*__regions`` by name with a built-in mapping."""

    MAPPING_FROM_OPTION = False
    PREFIX_PATTERN = r".*__regions$"

    @classmethod
    def resolve_mapping(cls, feature: Feature) -> dict[str, str]:
        return dict(_MAPPING)


class TestLookupIntegration(DataOpsIntegrationTestBase):
    @classmethod
    def feature_group_class(cls) -> type:
        return LookupFeatureGroup

    @classmethod
    def data_creator_class(cls) -> type:
        return PyArrowDataOpsTestDataCreator

    @classmethod
    def compute_framework_class(cls) -> type:
        return PyArrowTable

    @classmethod
    def primary_feature_name(cls) -> str:
        return "region__lookup"

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {"lookup_mapping": _MAPPING, "on_unmatched": "null"}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        return list(_REGION_NULL)

    @classmethod
    def secondary_feature_name(cls) -> str:
        return "region__lookup"

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        return {"lookup_mapping": _MAPPING, "on_unmatched": "keep"}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        return list(_REGION_KEEP)

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return ["region__lookup", "category__lookup"]

    @classmethod
    def invalid_feature_names(cls) -> list[str]:
        # The last guards against lookup stealing an ffill feature.
        return ["region", "region__lookup_extra", "lookup", "region__ffill"]

    @classmethod
    def match_options(cls) -> Options:
        return Options(context={"lookup_mapping": _MAPPING})


class TestLookupThroughMlodaApi:
    def _run(self, feature: Feature, name: str, *lookup_classes: type) -> pa.Table:
        enabled = {PyArrowDataOpsTestDataCreator, *(lookup_classes or (LookupFeatureGroup,))}
        plugin_collector = PluginCollector.enabled_feature_groups(enabled)
        results = mloda.run_all([feature], compute_frameworks=[PyArrowTable], plugin_collector=plugin_collector)
        tables = [t for t in results if isinstance(t, pa.Table) and name in t.column_names]
        assert tables, f"No result table with {name} found"
        return tables[0]

    def test_issue_syntax_with_dict_options(self) -> None:
        feature = Feature("region__lookup", options={"lookup_mapping": _MAPPING, "on_unmatched": "null"})
        result = self._run(feature, "region__lookup")
        assert result.column("region__lookup").to_pylist() == _REGION_NULL

    def test_unmatched_raise_is_reported(self) -> None:
        feature = Feature("region__lookup", options=Options(context={"lookup_mapping": _MAPPING}))
        with pytest.raises(LookupUnmatchedError, match=r"region__lookup: 3 rows, 1 distinct key not in the mapping"):
            self._run(feature, "region__lookup")

    def test_hook_subclass_end_to_end(self) -> None:
        feature = Feature("region__regions", options=Options(context={"on_unmatched": "null"}))
        result = self._run(feature, "region__regions", _RegionHookLookup)
        assert result.column("region__regions").to_pylist() == _REGION_NULL

    def test_keep_with_mismatched_families_fails_run(self) -> None:
        options = Options(context={"lookup_mapping": {"A": 1, "B": 2}, "on_unmatched": "keep"})
        with pytest.raises(ValueError, match="keep"):
            self._run(Feature("region__lookup", options=options), "region__lookup")

    def test_hook_feature_with_own_mapping_option_fails_run(self) -> None:
        options = Options(context={"lookup_mapping": _MAPPING})
        with pytest.raises(ValueError, match="lookup_mapping"):
            self._run(Feature("region__regions", options=options), "region__regions", _RegionHookLookup)

    def test_hook_feature_with_own_mapping_option_fails_run_with_generic_lookup_enabled(self) -> None:
        """The hook's own-mapping error wins over a multiple-match error when the generic class is enabled too."""
        options = Options(context={"lookup_mapping": _MAPPING, "in_features": "region"})
        with pytest.raises(ValueError, match="supplies its own mapping"):
            self._run(
                Feature("region__regions", options=options), "region__regions", LookupFeatureGroup, _RegionHookLookup
            )

    def test_hook_over_generic_lookup_with_inner_mapping(self) -> None:
        feature = Feature(
            "region__lookup__regions",
            options={"lookup_mapping": {"A": "B", "B": "A", "C": "A"}, "on_unmatched": "null"},
        )
        result = self._run(feature, "region__lookup__regions", LookupFeatureGroup, _RegionHookLookup)
        assert result.column("region__lookup__regions").to_pylist() == ["Beta"] * 4 + ["Alpha"] * 7 + [None]

    def test_generic_lookup_over_hook_with_group_mapping(self) -> None:
        feature = Feature(
            "region__regions__lookup",
            options={"lookup_mapping": {"Alpha": "a", "Beta": "b"}, "on_unmatched": "null"},
        )
        result = self._run(feature, "region__regions__lookup", LookupFeatureGroup, _RegionHookLookup)
        assert result.column("region__regions__lookup").to_pylist() == ["a"] * 4 + ["b"] * 4 + [None] * 4


class TestLookupSeveralFrameworks:
    """One LookupFeatureGroup is served by one Implementation per framework, chosen per feature."""

    @staticmethod
    def _collector() -> PluginCollector:
        return PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, LookupFeatureGroup})

    @staticmethod
    def _feature() -> Feature:
        return Feature("region__lookup", options={"lookup_mapping": _MAPPING, "on_unmatched": "null"})

    def test_several_frameworks_enabled_runs_lookup_and_names_the_implementation(self) -> None:
        pytest.importorskip("pandas")
        from mloda.core.abstract_plugins.implementation import loaded_implementations
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

        from mloda.community.feature_groups.data_operations.row_preserving.lookup.pandas_lookup import PandasLookup

        assert {PyArrowLookup, PandasLookup} <= set(loaded_implementations())
        frameworks = [PyArrowTable, PandasDataFrame]
        results = mloda.run_all([self._feature()], compute_frameworks=frameworks, plugin_collector=self._collector())
        tables = [t for t in results if isinstance(t, pa.Table) and "region__lookup" in t.column_names]
        assert len(tables) == 1
        assert tables[0].column("region__lookup").to_pylist() == _REGION_NULL

        plan = mloda.explain([self._feature()], compute_frameworks=frameworks, plugin_collector=self._collector())
        steps = [s for s in plan if s.step_kind == "compute" and s.feature_group is LookupFeatureGroup]
        assert len(steps) == 1
        assert steps[0].implementation is PyArrowLookup
        assert steps[0].compute_framework is PyArrowTable

    @staticmethod
    def _step(features: list[Feature], frameworks: list[Any], collector: PluginCollector, group: type) -> Any:
        plan = mloda.explain(features, compute_frameworks=frameworks, plugin_collector=collector)
        steps = [step for step in plan if step.step_kind == "compute" and step.feature_group is group]
        assert len(steps) == 1
        return steps[0]

    def test_pandas_source_runs_pandas_lookup_for_the_group_and_for_a_hook(self) -> None:
        pytest.importorskip("pandas")
        import pandas as pd
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

        from mloda.community.feature_groups.data_operations.row_preserving.lookup.pandas_lookup import PandasLookup

        frameworks = [PandasDataFrame, PyArrowTable]
        hook_feature = Feature("region__regions", options=Options(context={"on_unmatched": "null"}))
        for feature, name, group in (
            (self._feature(), "region__lookup", LookupFeatureGroup),
            (hook_feature, "region__regions", _RegionHookLookup),
        ):
            collector = PluginCollector.enabled_feature_groups(
                {PandasDataOpsTestDataCreator, LookupFeatureGroup, _RegionHookLookup}
            )
            step = self._step([feature], frameworks, collector, group)
            assert step.implementation is PandasLookup
            assert step.compute_framework is PandasDataFrame

            results = mloda.run_all([feature], compute_frameworks=frameworks, plugin_collector=collector)
            frames = [frame for frame in results if isinstance(frame, pd.DataFrame) and name in frame.columns]
            assert len(frames) == 1
            assert_values_with_nulls(frames[0][name].tolist(), _REGION_NULL, nan_is_null=True)

    @pytest.mark.parametrize("pin", ["qualified", "class"], ids=["by_qualified_name", "by_class"])
    def test_implementation_pin_runs_on_pandas(self, pin: str) -> None:
        """Feature(..., implementation=...) overrides the ranking and runs the pinned implementation."""
        pytest.importorskip("pandas")
        import pandas as pd
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

        from mloda.community.feature_groups.data_operations.row_preserving.lookup.pandas_lookup import PandasLookup

        qualified = "mloda.community.feature_groups.data_operations.row_preserving.lookup.pandas_lookup:PandasLookup"
        feature = Feature(
            "region__lookup",
            options={"lookup_mapping": _MAPPING, "on_unmatched": "null"},
            implementation=qualified if pin == "qualified" else PandasLookup,
        )
        frameworks = [PyArrowTable, PandasDataFrame]

        step = self._step([feature], frameworks, self._collector(), LookupFeatureGroup)
        assert step.implementation is PandasLookup
        assert step.compute_framework is PandasDataFrame

        results = mloda.run_all([feature], compute_frameworks=frameworks, plugin_collector=self._collector())
        frames = [frame for frame in results if isinstance(frame, pd.DataFrame)]
        assert len(frames) == 1
        assert_values_with_nulls(frames[0]["region__lookup"].tolist(), _REGION_NULL, nan_is_null=True)

    def test_group_without_an_enabled_implementation_is_a_resolution_error(self) -> None:
        collector = self._collector()
        collector.add_disabled_implementation_classes({PyArrowLookup})
        with pytest.raises(FeatureResolutionError) as excinfo:
            mloda.run_all([self._feature()], compute_frameworks=[PyArrowTable], plugin_collector=collector)
        assert excinfo.value.result.failure_kind == "none"
        assert "No feature groups found for feature name: 'region__lookup'" in str(excinfo.value)
        assert "LookupFeatureGroup (compute framework)" in str(excinfo.value)
        assert "disabled or denied" in str(excinfo.value)
