"""Runnable example of mloda's ASOF (point-in-time) join, linked from the as-of
join guide. It passes on first run and is exercised on both the pandas and
PyArrow backends (mloda 0.9.0 does PyArrow ASOF natively via Acero).

Backward join keyed by ``symbol``: each event picks the latest quote with
``quote_ts <= event_ts`` (exact match allowed). Expected price per event id::

    0 AAA@10:00 -> 100   1 AAA@10:15 -> 100   2 AAA@10:45 -> 110
    3 BBB@10:30 -> 220   4 BBB@11:00 -> 220
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import pytest

pd = pytest.importorskip("pandas")

# pandas is the baseline (always required); pyarrow is optional. Guard both the
# module and the compute-framework import so the pandas parametrization still
# runs when pyarrow is absent (only the pyarrow param is skipped, see below).
try:
    import pyarrow as pa

    from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
except ImportError:  # pragma: no cover - exercised only without pyarrow installed
    pa = None  # type: ignore[assignment]
    PyArrowTable = None  # type: ignore[assignment,misc]

from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda.provider import ComputeFramework, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Index, Link, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

_U = timezone.utc

# Expected price in effect at each event id (see module docstring for the trace).
_ASOF_BACKWARD_EXPECTED: dict[int, int] = {0: 100, 1: 100, 2: 110, 3: 220, 4: 220}


def _to_frame(framework: type[ComputeFramework], columns: dict[str, Any]) -> Any:
    """Build the native frame for ``framework`` (pandas ``DataFrame`` / PyArrow ``Table``)."""
    if framework is PandasDataFrame:
        return pd.DataFrame(columns)
    return pa.table(columns)


class _AsofEventSource(FeatureGroup):
    """LEFT side (events); subclasses only pin ``compute_framework``."""

    compute_framework: type[ComputeFramework]

    @classmethod
    def get_raw_data(cls) -> dict[str, list[Any]]:
        return {
            "id": [0, 1, 2, 3, 4],
            "symbol": ["AAA", "AAA", "AAA", "BBB", "BBB"],
            "event_ts": [
                datetime(2023, 1, 1, 10, 0, 0, tzinfo=_U),
                datetime(2023, 1, 1, 10, 15, 0, tzinfo=_U),
                datetime(2023, 1, 1, 10, 45, 0, tzinfo=_U),
                datetime(2023, 1, 1, 10, 30, 0, tzinfo=_U),
                datetime(2023, 1, 1, 11, 0, 0, tzinfo=_U),
            ],
        }

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("symbol",))]

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(set(cls.get_raw_data().keys()))

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _to_frame(cls.compute_framework, cls.get_raw_data())

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {cls.compute_framework}


class _AsofQuoteSource(FeatureGroup):
    """RIGHT side (per-symbol price timeline); subclasses only pin ``compute_framework``."""

    compute_framework: type[ComputeFramework]

    @classmethod
    def get_raw_data(cls) -> dict[str, list[Any]]:
        return {
            "symbol": ["AAA", "AAA", "BBB", "BBB"],
            "quote_ts": [
                datetime(2023, 1, 1, 10, 0, 0, tzinfo=_U),
                datetime(2023, 1, 1, 10, 30, 0, tzinfo=_U),
                datetime(2023, 1, 1, 10, 0, 0, tzinfo=_U),
                datetime(2023, 1, 1, 10, 30, 0, tzinfo=_U),
            ],
            "price": [100, 110, 200, 220],
        }

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("symbol",))]

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(set(cls.get_raw_data().keys()))

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _to_frame(cls.compute_framework, cls.get_raw_data())

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {cls.compute_framework}


class _AsofEventPrice(FeatureGroup):
    """Consumer that pulls from both sources (triggering the ASOF link) and exposes
    ``asof_event_id`` / ``asof_event_price`` so the test can assert keyed by id."""

    compute_framework: type[ComputeFramework]

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"asof_event_id", "asof_event_price"}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        # id + event_ts resolve to the event source; quote_ts + price to the quote source.
        # symbol (the by-key) is retained automatically as the index column.
        return {
            Feature("id"),
            Feature("event_ts"),
            Feature("quote_ts"),
            Feature("price"),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _to_frame(cls.compute_framework, {"asof_event_id": data["id"], "asof_event_price": data["price"]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {cls.compute_framework}


class AsofEventSourcePandas(_AsofEventSource):
    compute_framework: type[ComputeFramework] = PandasDataFrame


class AsofQuoteSourcePandas(_AsofQuoteSource):
    compute_framework: type[ComputeFramework] = PandasDataFrame


class AsofEventPricePandas(_AsofEventPrice):
    compute_framework: type[ComputeFramework] = PandasDataFrame


class AsofEventSourcePyArrow(_AsofEventSource):
    compute_framework: type[ComputeFramework] = PyArrowTable


class AsofQuoteSourcePyArrow(_AsofQuoteSource):
    compute_framework: type[ComputeFramework] = PyArrowTable


class AsofEventPricePyArrow(_AsofEventPrice):
    compute_framework: type[ComputeFramework] = PyArrowTable


def _extract_asof_prices(results: list[Any], framework: type[ComputeFramework]) -> dict[int, int]:
    """Read ``{id: price}`` from the result, asserting it is ``framework``'s native
    frame type so a wrong-backend result (e.g. a pandas fallback) fails loudly."""
    for table in results:
        if isinstance(table, pd.DataFrame) and "asof_event_price" in table.columns:
            assert framework is PandasDataFrame, f"Expected a {framework.__name__} result frame, got pandas.DataFrame"
            return {int(i): int(p) for i, p in zip(table["asof_event_id"], table["asof_event_price"])}
        if pa is not None and isinstance(table, pa.Table) and "asof_event_price" in table.column_names:
            assert framework is PyArrowTable, f"Expected a {framework.__name__} result frame, got pyarrow.Table"
            ids = table.column("asof_event_id").to_pylist()
            prices = table.column("asof_event_price").to_pylist()
            return {int(i): int(p) for i, p in zip(ids, prices)}

    raise AssertionError(f"No native {framework.__name__} result frame with asof_event_price found")


def _run_asof_backward(
    framework: type[ComputeFramework],
    event_source: type[FeatureGroup],
    quote_source: type[FeatureGroup],
    event_price: type[FeatureGroup],
) -> dict[int, int]:
    """Run the backward ASOF join through ``run_all`` and return ``{id: price}``."""
    link = Link.asof_on(
        event_source,
        quote_source,
        left_time_column="event_ts",
        right_time_column="quote_ts",
        direction="backward",
    )
    plugin_collector = PluginCollector.enabled_feature_groups({event_source, quote_source, event_price})

    results = mloda.run_all(
        [Feature("asof_event_id"), Feature("asof_event_price")],
        compute_frameworks={framework},
        links={link},
        plugin_collector=plugin_collector,
    )

    return _extract_asof_prices(results, framework)


class TestAsofJoinExample:
    """Backward point-in-time join happy path on the pandas and PyArrow backends."""

    @pytest.mark.parametrize(
        "framework, event_source, quote_source, event_price",
        [
            pytest.param(
                PandasDataFrame,
                AsofEventSourcePandas,
                AsofQuoteSourcePandas,
                AsofEventPricePandas,
                id="pandas",
            ),
            pytest.param(
                PyArrowTable,
                AsofEventSourcePyArrow,
                AsofQuoteSourcePyArrow,
                AsofEventPricePyArrow,
                id="pyarrow",
                marks=pytest.mark.skipif(pa is None or PyArrowTable is None, reason="pyarrow not installed"),
            ),
        ],
    )
    def test_backward_asof_picks_price_in_effect(
        self,
        framework: type[ComputeFramework],
        event_source: type[FeatureGroup],
        quote_source: type[FeatureGroup],
        event_price: type[FeatureGroup],
    ) -> None:
        result = _run_asof_backward(framework, event_source, quote_source, event_price)
        assert result == _ASOF_BACKWARD_EXPECTED
