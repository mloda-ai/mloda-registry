"""Integration tests for sessionization through mloda's full pipeline.

Verifies that the sessionization feature group resolves through
``mloda.run_all`` with a ``PluginCollector``, including plugin discovery,
feature-name resolution (``match_feature_group_criteria``), and pipeline
plumbing.

sessionization computes natively on every backend. The shared
``DataOpsIntegrationTestBase`` runs on a small pandas-only data source (the
canonical 12-row dataset has a null and duplicate ``timestamp`` values that make
a by-hand session-id pin awkward) whose expected session ids are computed by hand
from the documented algorithm (gap > threshold starts a new session;
``session_id = cumsum(is_new) - 1``). ``OperationGroupIntegrationTestMixin`` adds
the operation-group checks. Matching is name-based, so
``test_match_rejects_missing_config`` is overridden to use a non-pattern name.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from mloda.provider import ComputeFramework, DataCreatorFeatureGroup, FeatureSet
from mloda.user import Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

# Loads the Implementations that serve the group.
from mloda.community.feature_groups.data_operations.row_preserving.sessionization import (
    manifest as _manifest,  # noqa: F401
)
from mloda.community.feature_groups.data_operations.row_preserving.sessionization.base import (
    SessionizationFeatureGroup,
)
from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase
from mloda.testing.feature_groups.data_operations.mixins.operation_group_integration import (
    OperationGroupIntegrationTestMixin,
)

_U = timezone.utc

# Single user, whole-table stream. Gaps in time order:
#   10:00 (new s0) -> 10:25 (+25<=30 same s0) -> 11:00 (+35>30 new s1)
#   -> 11:20 (+20<=30 same s1) -> 12:30 (+70>30 new s2)
_SESSION_30_MINUTE_EXPECTED: list[int] = [0, 0, 1, 1, 2]
# Every gap exceeds the 10 minute threshold, so each row is its own session.
_SESSION_10_MINUTE_EXPECTED: list[int] = [0, 1, 2, 3, 4]


class SessionIntegrationDataCreator(DataCreatorFeatureGroup):
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
    def feature_names_supported(cls) -> set[str]:
        return set(cls.get_raw_data().keys())

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame(cls.get_raw_data())

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {cls.compute_framework}


class TestSessionizationIntegration(OperationGroupIntegrationTestMixin, DataOpsIntegrationTestBase):
    @classmethod
    def feature_group_class(cls) -> type:
        return SessionizationFeatureGroup

    @classmethod
    def data_creator_class(cls) -> type:
        return SessionIntegrationDataCreator

    @classmethod
    def compute_framework_class(cls) -> type:
        return PandasDataFrame

    @classmethod
    def several_frameworks(cls) -> list[type[Any]]:
        return [PandasDataFrame, PyArrowTable]

    @classmethod
    def expected_row_count(cls) -> int:
        return 5

    @classmethod
    def primary_feature_name(cls) -> str:
        return "ts__sessionize_30_minute"

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {"order_by": "ts", "partition_by": ["user"]}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        return _SESSION_30_MINUTE_EXPECTED

    @classmethod
    def secondary_feature_name(cls) -> str:
        return "ts__sessionize_10_minute"

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        return {"order_by": "ts", "partition_by": ["user"]}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        return _SESSION_10_MINUTE_EXPECTED

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return ["ts__sessionize_30_minute", "event__sessionize_1_hour"]

    @classmethod
    def invalid_feature_names(cls) -> list[str]:
        return ["ts", "ts__sessionize", "ts__sessionize_30", "ts__sessionize_30_month"]

    @classmethod
    def match_options(cls) -> Options:
        return Options(context={"order_by": "ts", "partition_by": ["user"]})

    def test_match_rejects_missing_config(self) -> None:
        """Name-based: a non-pattern name without config must fail to match."""
        assert not SessionizationFeatureGroup.match_feature_group_criteria("my_custom_result", Options())
