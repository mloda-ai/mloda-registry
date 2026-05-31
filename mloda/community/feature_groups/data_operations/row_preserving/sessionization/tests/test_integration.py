"""Integration tests for sessionization through mloda's full pipeline.

Verifies that the sessionization feature group resolves through
``mloda.run_all`` with a ``PluginCollector``, including plugin discovery,
feature-name resolution (``match_feature_group_criteria``), and pipeline
plumbing.

sessionization computes natively on every backend, but the shared
``DataOpsIntegrationTestBase`` is tied to the canonical 12-row dataset, whose
``timestamp`` column has a null and duplicate values that make a by-hand
session-id pin awkward. This module therefore uses bespoke ``run_all`` tests
(mirroring ema's ``TestEmaIntegration``) with a small pandas-only data source
whose expected session ids are computed by hand from the documented algorithm
(gap > threshold starts a new session; ``session_id = cumsum(is_new) - 1``).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import ComputeFramework, FeatureGroup, FeatureSet
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.sessionization.pandas_sessionization import (
    PandasSessionization,
)

_U = timezone.utc

# Single user, whole-table stream. Gaps in time order:
#   10:00 (new s0) -> 10:25 (+25<=30 same s0) -> 11:00 (+35>30 new s1)
#   -> 11:20 (+20<=30 same s1) -> 12:30 (+70>30 new s2)
_SESSION_30_MINUTE_EXPECTED: list[int] = [0, 0, 1, 1, 2]


class SessionIntegrationDataCreator(FeatureGroup):
    """Local pandas data source: one user, five rows with deliberate gaps."""

    compute_framework: type[ComputeFramework] = PandasDataFrame

    @classmethod
    def get_raw_data(cls) -> dict[str, list[Any]]:
        return {
            "id": [0, 1, 2, 3, 4],
            "user": ["A", "A", "A", "A", "A"],
            "ts": [
                datetime(2023, 1, 1, 10, 0, 0, tzinfo=_U),
                datetime(2023, 1, 1, 10, 25, 0, tzinfo=_U),
                datetime(2023, 1, 1, 11, 0, 0, tzinfo=_U),
                datetime(2023, 1, 1, 11, 20, 0, tzinfo=_U),
                datetime(2023, 1, 1, 12, 30, 0, tzinfo=_U),
            ],
        }

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(set(cls.get_raw_data().keys()))

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame(cls.get_raw_data())

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {cls.compute_framework}


def _session_values(name: str, context: dict[str, Any]) -> list[Any]:
    """Run a single sessionization feature on pandas through ``run_all``."""
    plugin_collector = PluginCollector.enabled_feature_groups({SessionIntegrationDataCreator, PandasSessionization})
    feature = Feature(name, options=Options(context=context))

    results = mloda.run_all(
        [feature],
        compute_frameworks={PandasDataFrame},
        plugin_collector=plugin_collector,
    )

    for table in results:
        if isinstance(table, pd.DataFrame) and name in table.columns:
            return list(table[name])

    raise AssertionError(f"No result frame with {name} found")


class TestSessionizationIntegration:
    """Bespoke ``run_all`` integration tests on the pandas backend."""

    def test_sessionize_30_minute_through_pipeline(self) -> None:
        values = _session_values(
            "ts__sessionize_30_minute",
            {"order_by": "ts", "partition_by": ["user"]},
        )
        assert len(values) == 5
        assert [int(v) for v in values] == _SESSION_30_MINUTE_EXPECTED


class TestSessionizationMatchFeatureGroupCriteria:
    """match_feature_group_criteria routing for the sessionize pattern."""

    def test_valid_names_match(self) -> None:
        options = Options(context={"order_by": "ts", "partition_by": ["user"]})
        for name in ["ts__sessionize_30_minute", "event__sessionize_1_hour"]:
            assert PandasSessionization.match_feature_group_criteria(name, options), f"expected {name} to match"

    def test_invalid_names_do_not_match(self) -> None:
        options = Options(context={"order_by": "ts", "partition_by": ["user"]})
        for name in ["ts", "ts__sessionize", "ts__sessionize_30", "ts__sessionize_30_month"]:
            assert not PandasSessionization.match_feature_group_criteria(name, options), f"expected {name} to NOT match"
