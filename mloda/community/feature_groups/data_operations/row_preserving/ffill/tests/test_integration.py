"""Integration tests for ffill through mloda's full pipeline.

Verifies that the forward-fill feature group resolves through ``mloda.run_all``
with a ``PluginCollector``, including plugin discovery, feature-name resolution
(``match_feature_group_criteria``), and pipeline plumbing.

ffill is row-preserving and supports PyArrow natively, so the standard
``DataOpsIntegrationTestBase`` (12-row, exact column compare on PyArrow tables)
fits directly. Matching is name-based (the ``.*__ffill$`` pattern), so
``test_match_rejects_missing_config`` is overridden to use a non-pattern name.

Expected fill columns are computed OFFLINE on the canonical dataset (per
partition, in timestamp order) and pinned as literals; the production feature
group is not imported to generate its own oracle.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.ffill.pyarrow_ffill import (
    PyArrowFfill,
)

# Forward-fill of ``value_float`` within each ``region`` partition, in timestamp
# order. Region A: [1.5, 2.5, None->2.5, 0.0]; B: [-3.14, 5.5, 6.5, 7.5];
# C: [1e-15, 100.0, 0.0]; None-region: [None] (leading null stays null).
_VALUE_FLOAT_FFILL_BY_REGION: list[Any] = [1.5, 2.5, 2.5, 0.0, -3.14, 5.5, 6.5, 7.5, 1e-15, 100.0, 0.0, None]

# Forward-fill of ``value_float`` over the WHOLE table (no partition), in
# timestamp order. The null-timestamp row (index 10) sorts last; the final row
# (index 11) carries the prior non-null 100.0 forward.
_VALUE_FLOAT_FFILL_WHOLE: list[Any] = [1.5, 2.5, 2.5, 0.0, -3.14, 5.5, 6.5, 7.5, 1e-15, 100.0, 0.0, 100.0]

# Forward-fill of ``amount`` within each ``region`` partition, in timestamp order.
_AMOUNT_FFILL_BY_REGION: list[Any] = [100.0, 100.0, 250.0, 75.0, 300.0, 0.0, 150.0, 150.0, 50.0, 200.0, 125.0, 80.0]


class TestFfillIntegration(DataOpsIntegrationTestBase):
    """Standard integration tests inherited from the base class.

    ffill uses name-based matching, so ``match_feature_group_criteria`` succeeds
    with empty Options when the feature name matches the ``.*__ffill$`` pattern.
    ``test_match_rejects_missing_config`` is overridden below to use a non-pattern
    name that cannot match without config.
    """

    @classmethod
    def feature_group_class(cls) -> type:
        return PyArrowFfill

    @classmethod
    def data_creator_class(cls) -> type:
        return PyArrowDataOpsTestDataCreator

    @classmethod
    def compute_framework_class(cls) -> type:
        return PyArrowTable

    @classmethod
    def primary_feature_name(cls) -> str:
        return "value_float__ffill"

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {"order_by": "timestamp", "partition_by": ["region"]}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        return list(_VALUE_FLOAT_FFILL_BY_REGION)

    @classmethod
    def secondary_feature_name(cls) -> str:
        return "value_float__ffill"

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        # No partition_by: whole table is a single partition.
        return {"order_by": "timestamp"}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        return list(_VALUE_FLOAT_FFILL_WHOLE)

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return ["value_float__ffill", "amount__ffill"]

    @classmethod
    def invalid_feature_names(cls) -> list[str]:
        # The last guards against ffill stealing an ema feature.
        return ["value_float", "value_float__ffill_extra", "ffill", "value_float__ema_2"]

    @classmethod
    def match_options(cls) -> Options:
        return Options(context={"order_by": "timestamp", "partition_by": ["region"]})

    def test_match_rejects_missing_config(self) -> None:
        """Name-based: a non-pattern name without config must fail to match."""
        options = Options()
        assert not PyArrowFfill.match_feature_group_criteria("my_custom_result", options)


class TestIntegrationMultipleFeatures:
    """Run multiple ffill features in a single ``run_all`` call."""

    def test_two_ffill_features_together(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, PyArrowFfill})

        f_value = Feature(
            "value_float__ffill",
            options=Options(context={"order_by": "timestamp", "partition_by": ["region"]}),
        )
        f_amount = Feature(
            "amount__ffill",
            options=Options(context={"order_by": "timestamp", "partition_by": ["region"]}),
        )

        results = mloda.run_all(
            [f_value, f_amount],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        value_found = False
        amount_found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "value_float__ffill" in table.column_names:
                col = table.column("value_float__ffill").to_pylist()
                assert col == _VALUE_FLOAT_FFILL_BY_REGION
                value_found = True
            if "amount__ffill" in table.column_names:
                col = table.column("amount__ffill").to_pylist()
                assert col == _AMOUNT_FFILL_BY_REGION
                amount_found = True

        assert value_found, "value_float__ffill result not found"
        assert amount_found, "amount__ffill result not found"
