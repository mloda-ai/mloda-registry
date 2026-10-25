"""Integration tests for EMA through mloda's full pipeline.

Verifies that the exponential-moving-average feature group resolves through
``mloda.run_all`` with a ``PluginCollector``, including plugin discovery,
feature-name resolution (``match_feature_group_criteria``), and pipeline
plumbing.

EMA computes natively only on pandas / polars-lazy; it ships NO PyArrow (or
duckdb / sqlite) backend at all (the absence convention: a recursive Python
emulation is forbidden by the CFW-backend rule). The shared
``DataOpsIntegrationTestBase`` therefore runs on pandas with a pandas data
creator, and ``OperationGroupIntegrationTestMixin`` adds the operation-group checks.

Expected EMA columns are computed OFFLINE on the canonical dataset using the
pinned formula ``s.ewm(span=span, adjust=False, ignore_na=True).mean().mask(
s.isna())`` per region in timestamp order, and pinned as literals; the
production feature group is not imported to generate its own oracle.
"""

from __future__ import annotations

from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from mloda.user import Feature, FeatureResolutionError, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

# Loads the Implementations that serve the group.
from mloda.community.feature_groups.data_operations.row_preserving.ema import manifest as _manifest  # noqa: F401
from mloda.community.feature_groups.data_operations.row_preserving.ema.base import EmaFeatureGroup
from mloda.testing.data_creator.pandas import PandasDataOpsTestDataCreator
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase
from mloda.testing.feature_groups.data_operations.mixins.operation_group_integration import (
    OperationGroupIntegrationTestMixin,
)

# EMA of ``value_float`` per ``region`` in timestamp order, adjust=False,
# nulls skipped (ignore_na=True), output null where the input is null.
_EMA_2_BY_REGION: list[Any] = [
    1.5,
    2.1666666666666665,
    None,
    0.7222222222222222,
    -3.14,
    2.6199999999999997,
    5.206666666666666,
    6.735555555555555,
    1e-15,
    66.66666666666666,
    22.22222222222222,
    None,
]
_EMA_3_BY_REGION: list[Any] = [
    1.5,
    2.0,
    None,
    1.0,
    -3.14,
    1.18,
    3.84,
    5.67,
    1e-15,
    50.0,
    25.0,
    None,
]


class TestEmaIntegration(OperationGroupIntegrationTestMixin, DataOpsIntegrationTestBase):
    @classmethod
    def feature_group_class(cls) -> type:
        return EmaFeatureGroup

    @classmethod
    def data_creator_class(cls) -> type:
        return PandasDataOpsTestDataCreator

    @classmethod
    def compute_framework_class(cls) -> type:
        return PandasDataFrame

    @classmethod
    def several_frameworks(cls) -> list[type[Any]]:
        pytest.importorskip("polars")
        from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

        return [PandasDataFrame, PolarsLazyDataFrame]

    @classmethod
    def primary_feature_name(cls) -> str:
        return "value_float__ema_2"

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {"order_by": "timestamp", "partition_by": ["region"]}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        return _EMA_2_BY_REGION

    @classmethod
    def secondary_feature_name(cls) -> str:
        return "value_float__ema_3"

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        return {"order_by": "timestamp", "partition_by": ["region"]}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        return _EMA_3_BY_REGION

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return ["value_float__ema_2", "amount__ema_5"]

    @classmethod
    def invalid_feature_names(cls) -> list[str]:
        # ema_0 matches the regex and is rejected at compute time instead (see TestEmaRejectionRouting).
        return ["value_float", "value_float__ema", "ema_2", "value_float__ffill"]

    @classmethod
    def match_options(cls) -> Options:
        return Options(context={"order_by": "timestamp", "partition_by": ["region"]})

    @classmethod
    def use_approx(cls) -> bool:
        return True

    @classmethod
    def approx_rel(cls) -> float:
        return 1e-9

    @classmethod
    def nan_is_null(cls) -> bool:
        return True

    def test_match_rejects_missing_config(self) -> None:
        """Name-based: a non-pattern name without config must fail to match."""
        assert not EmaFeatureGroup.match_feature_group_criteria("my_custom_result", Options())


class TestEmaMatchFeatureGroupCriteria:
    """match_feature_group_criteria cases the shared base does not carry."""

    def test_missing_order_by_does_not_match(self) -> None:
        """Name path: a pattern name without order_by is a non-match."""
        options = Options(context={"partition_by": ["region"]})
        assert not EmaFeatureGroup.match_feature_group_criteria("value_float__ema_2", options)


class TestEmaRejectionRouting:
    """EMA must fail loudly rather than emulate, via two distinct paths.

    ``test_ema_0_rejected_at_compute`` covers a per-input compute rejection
    (span=0 matches the name pattern but is rejected inside the working pandas
    backend). ``test_ema_no_emulation_on_pyarrow`` covers the absence path: EMA
    ships no pyarrow backend, so a request pinned to PyArrow fails to resolve
    with a FeatureResolutionError. Neither path emulates.
    """

    def test_ema_0_rejected_at_compute(self) -> None:
        """span=0 matches the name pattern but is rejected when computed."""
        plugin_collector = PluginCollector.enabled_feature_groups({PandasDataOpsTestDataCreator, EmaFeatureGroup})
        feature = Feature(
            "value_float__ema_0",
            options=Options(context={"order_by": "timestamp", "partition_by": ["region"]}),
        )

        with pytest.raises(Exception, match="span"):
            mloda.run_all(
                [feature],
                compute_frameworks=[PandasDataFrame],
                plugin_collector=plugin_collector,
            )

    def test_ema_no_emulation_on_pyarrow(self) -> None:
        """EMA on PyArrow must fail loudly (no Python emulation) via ABSENCE.

        EMA computes natively only on pandas / polars-lazy. Under the absence
        convention there is NO pyarrow (or duckdb / sqlite) EMA backend class at
        all -- a recursive Python emulation is forbidden by the CFW-backend rule.
        So even with a PyArrow data creator and the working pandas backend both
        enabled, a request pinned to ``compute_frameworks=[PyArrowTable]`` cannot
        resolve to any EMA implementation and mloda core raises a
        FeatureResolutionError. It does NOT silently emulate EMA.
        """
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, EmaFeatureGroup})
        feature = Feature(
            "value_float__ema_2",
            options=Options(context={"order_by": "timestamp", "partition_by": ["region"]}),
        )

        with pytest.raises(
            FeatureResolutionError,
            match=r"EmaFeatureGroup \(compute framework\): none of its compute frameworks are enabled for this run",
        ):
            mloda.run_all(
                [feature],
                compute_frameworks=[PyArrowTable],
                plugin_collector=plugin_collector,
            )
