"""Integration tests for EMA through mloda's full pipeline.

Verifies that the exponential-moving-average feature group resolves through
``mloda.run_all`` with a ``PluginCollector``, including plugin discovery,
feature-name resolution (``match_feature_group_criteria``), and pipeline
plumbing.

EMA computes natively only on pandas / polars-lazy; it ships NO PyArrow (or
duckdb / sqlite) backend at all (the absence convention: a recursive Python
emulation is forbidden by the CFW-backend rule). The shared
``DataOpsIntegrationTestBase`` is PyArrow-only (its result filter and value
assertions use ``pa.Table`` APIs), so it does NOT fit a pandas-only FG. This
module therefore uses bespoke ``run_all`` tests (mirroring time_bucketization's
``TestIntegrationMultipleFeatures``) with a local pandas data creator.

Expected EMA columns are computed OFFLINE on the canonical dataset using the
pinned formula ``s.ewm(span=span, adjust=False, ignore_na=True).mean().mask(
s.isna())`` per region in timestamp order, and pinned as literals; the
production feature group is not imported to generate its own oracle.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.base import DataOperationsTestDataCreator
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.ema.pandas_ema import (
    PandasEma,
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


class PandasDataOpsTestDataCreator(DataOperationsTestDataCreator):
    """Local pandas data source for the canonical 12-row dataset.

    EMA has no PyArrow backend, so the PyArrow data creator cannot feed the
    compute test. This test-only creator returns the canonical data as a pandas
    DataFrame.
    """

    compute_framework = PandasDataFrame

    @classmethod
    def create(cls) -> Any:
        return pd.DataFrame(cls.get_raw_data())


def _ema_values(name: str, context: dict[str, Any]) -> list[Any]:
    """Run a single EMA feature on pandas through ``run_all`` and return its column."""
    plugin_collector = PluginCollector.enabled_feature_groups({PandasDataOpsTestDataCreator, PandasEma})
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


def _assert_ema_equal(actual: list[Any], expected: list[Any]) -> None:
    """Compare EMA columns elementwise, treating NaN and None as equal nulls."""
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        got_is_null = got is None or (isinstance(got, float) and math.isnan(got))
        want_is_null = want is None
        if want_is_null:
            assert got_is_null, f"expected null, got {got!r}"
        else:
            assert not got_is_null
            assert got == pytest.approx(want, rel=1e-9)


class TestEmaIntegration:
    """Bespoke ``run_all`` integration tests for EMA on the pandas backend."""

    def test_ema_2_through_pipeline(self) -> None:
        values = _ema_values(
            "value_float__ema_2",
            {"order_by": "timestamp", "partition_by": ["region"]},
        )
        assert len(values) == 12
        _assert_ema_equal(values, _EMA_2_BY_REGION)

    def test_ema_3_through_pipeline(self) -> None:
        values = _ema_values(
            "value_float__ema_3",
            {"order_by": "timestamp", "partition_by": ["region"]},
        )
        assert len(values) == 12
        _assert_ema_equal(values, _EMA_3_BY_REGION)


class TestEmaMatchFeatureGroupCriteria:
    """match_feature_group_criteria routing for the ``.*__ema_\\d+$`` pattern."""

    def test_valid_names_match(self) -> None:
        options = Options(context={"order_by": "timestamp", "partition_by": ["region"]})
        for name in ["value_float__ema_2", "amount__ema_5"]:
            assert PandasEma.match_feature_group_criteria(name, options), f"expected {name} to match"

    def test_invalid_names_do_not_match(self) -> None:
        options = Options(context={"order_by": "timestamp", "partition_by": ["region"]})
        # ema_0 is handled separately: the regex accepts the digit, so it MATCHES
        # here and is rejected at compute time instead (see test below).
        for name in ["value_float", "value_float__ema", "ema_2", "value_float__ffill"]:
            assert not PandasEma.match_feature_group_criteria(name, options), f"expected {name} to NOT match"

    def test_match_rejects_missing_config(self) -> None:
        """Name-based: a non-pattern name without config must fail to match."""
        assert not PandasEma.match_feature_group_criteria("my_custom_result", Options())


class TestEmaRejectionRouting:
    """EMA must fail loudly rather than emulate, via two distinct paths.

    ``test_ema_0_rejected_at_compute`` covers a per-input compute rejection
    (span=0 matches the name pattern but is rejected inside the working pandas
    backend). ``test_ema_no_emulation_on_pyarrow`` covers the absence path: EMA
    ships no pyarrow backend, so a request pinned to PyArrow fails to resolve
    with mloda core's generic no-feature-group error. Neither path emulates.
    """

    def test_ema_0_rejected_at_compute(self) -> None:
        """span=0 matches the name pattern but is rejected when computed."""
        plugin_collector = PluginCollector.enabled_feature_groups({PandasDataOpsTestDataCreator, PandasEma})
        feature = Feature(
            "value_float__ema_0",
            options=Options(context={"order_by": "timestamp", "partition_by": ["region"]}),
        )

        with pytest.raises(Exception, match="span"):
            mloda.run_all(
                [feature],
                compute_frameworks={PandasDataFrame},
                plugin_collector=plugin_collector,
            )

    def test_ema_no_emulation_on_pyarrow(self) -> None:
        """EMA on PyArrow must fail loudly (no Python emulation) via ABSENCE.

        EMA computes natively only on pandas / polars-lazy. Under the absence
        convention there is NO pyarrow (or duckdb / sqlite) EMA backend class at
        all -- a recursive Python emulation is forbidden by the CFW-backend rule.
        So even with a PyArrow data creator and the working pandas backend both
        enabled, a request pinned to ``compute_frameworks={PyArrowTable}`` cannot
        resolve to any EMA backend and mloda core raises its generic
        no-feature-group resolution error. It does NOT silently emulate EMA.
        """
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, PandasEma})
        feature = Feature(
            "value_float__ema_2",
            options=Options(context={"order_by": "timestamp", "partition_by": ["region"]}),
        )

        with pytest.raises(ValueError, match=r"(?i)no feature group"):
            mloda.run_all(
                [feature],
                compute_frameworks={PyArrowTable},
                plugin_collector=plugin_collector,
            )
