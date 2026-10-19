"""Integration tests for lookup through mloda's full pipeline (PyArrow).

Expected columns are pinned as literals on the canonical dataset, whose ``region``
column is A x4, B x4, C x3, None x1.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest
from mloda.user import Feature, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.lookup.pyarrow_lookup import PyArrowLookup
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase

_MAPPING: dict[str, str] = {"A": "Alpha", "B": "Beta"}

_REGION_NULL: list[Any] = ["Alpha"] * 4 + ["Beta"] * 4 + [None] * 4
_REGION_KEEP: list[Any] = ["Alpha"] * 4 + ["Beta"] * 4 + ["C"] * 3 + [None]


class _RegionHookLookup(PyArrowLookup):
    """Hook subclass: matches ``*__regions`` by name with a built-in mapping."""

    MAPPING_FROM_OPTION = False
    PREFIX_PATTERN = r".*__regions$"

    @classmethod
    def resolve_mapping(cls, feature: Feature) -> dict[str, str]:
        return dict(_MAPPING)


class TestLookupIntegration(DataOpsIntegrationTestBase):
    @classmethod
    def feature_group_class(cls) -> type:
        return PyArrowLookup

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
    def _run(self, feature: Feature, name: str, lookup_class: type = PyArrowLookup) -> pa.Table:
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, lookup_class})
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
        with pytest.raises(Exception, match=r"region__lookup: 3 rows, 1 distinct keys not in the mapping"):
            self._run(feature, "region__lookup")

    def test_hook_subclass_end_to_end(self) -> None:
        feature = Feature("region__regions", options=Options(context={"on_unmatched": "null"}))
        result = self._run(feature, "region__regions", _RegionHookLookup)
        assert result.column("region__regions").to_pylist() == _REGION_NULL
