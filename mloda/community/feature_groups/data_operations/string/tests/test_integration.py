"""Integration tests for string operations through mloda's full pipeline.

These tests verify that string operation feature groups work end-to-end
through mloda's runtime, including plugin discovery, feature resolution,
and the PluginCollector mechanism.

Uses the shared DataOpsIntegrationTestBase from the testing library,
following the same pattern as the window_aggregation and datetime
integration tests.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.string.pyarrow_string import (
    PyArrowStringOps,
)


class TestStringIntegration(DataOpsIntegrationTestBase):
    """Standard integration tests inherited from the base class.

    String operations use pattern-based matching (``<source>__<op>``),
    so ``match_feature_group_criteria`` succeeds with empty Options when
    the feature name contains the pattern. The ``test_match_rejects_missing_config``
    test is overridden to use a non-pattern name that requires config-based matching.
    """

    @classmethod
    def feature_group_class(cls) -> type:
        return PyArrowStringOps

    @classmethod
    def data_creator_class(cls) -> type:
        return PyArrowDataOpsTestDataCreator

    @classmethod
    def compute_framework_class(cls) -> type:
        return PyArrowTable

    @classmethod
    def primary_feature_name(cls) -> str:
        return "name__upper"

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        return ["ALICE", "BOB", None, "", " EVE ", "FRANK", "GRACE", "ALICE", "  ", "BOB", "H\u00c9LLO", None]

    @classmethod
    def secondary_feature_name(cls) -> str:
        return "name__length"

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        return [5, 3, None, 0, 5, 5, 5, 5, 2, 3, 5, None]

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return [
            "name__upper",
            "name__lower",
            "name__trim",
            "name__length",
            "name__reverse",
        ]

    @classmethod
    def invalid_feature_names(cls) -> list[str]:
        return ["name", "name__split", "upper", "name__sum_groupby"]

    @classmethod
    def match_options(cls) -> Options:
        return Options()

    def test_match_rejects_missing_config(self) -> None:
        """String uses pattern-based matching. A non-pattern name without config should fail."""
        options = Options()
        assert not PyArrowStringOps.match_feature_group_criteria("my_custom_result", options)


class TestIntegrationMultipleFeatures:
    """Test multiple string operation features in a single run_all call."""

    def test_upper_and_lower_together(self) -> None:
        """Request both upper and lower features in one pipeline run."""
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, PyArrowStringOps})

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
                assert upper_col == [
                    "ALICE",
                    "BOB",
                    None,
                    "",
                    " EVE ",
                    "FRANK",
                    "GRACE",
                    "ALICE",
                    "  ",
                    "BOB",
                    "H\u00c9LLO",
                    None,
                ]
                upper_found = True
            if "name__lower" in table.column_names:
                lower_col = table.column("name__lower").to_pylist()
                assert lower_col == [
                    "alice",
                    "bob",
                    None,
                    "",
                    " eve ",
                    "frank",
                    "grace",
                    "alice",
                    "  ",
                    "bob",
                    "h\u00e9llo",
                    None,
                ]
                lower_found = True

        assert upper_found, "name__upper result not found in any result table"
        assert lower_found, "name__lower result not found in any result table"

    def test_trim_and_length_together(self) -> None:
        """Request trim and length features in one pipeline run."""
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, PyArrowStringOps})

        f_trim = Feature("name__trim", options=Options())
        f_length = Feature("name__length", options=Options())

        results = mloda.run_all(
            [f_trim, f_length],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        trim_found = False
        length_found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "name__trim" in table.column_names:
                trim_col = table.column("name__trim").to_pylist()
                assert trim_col == [
                    "Alice",
                    "bob",
                    None,
                    "",
                    "Eve",
                    "FRANK",
                    "Grace",
                    "alice",
                    "",
                    "Bob",
                    "h\u00e9llo",
                    None,
                ]
                trim_found = True
            if "name__length" in table.column_names:
                length_col = table.column("name__length").to_pylist()
                assert length_col == [5, 3, None, 0, 5, 5, 5, 5, 2, 3, 5, None]
                length_found = True

        assert trim_found, "name__trim result not found in any result table"
        assert length_found, "name__length result not found in any result table"
