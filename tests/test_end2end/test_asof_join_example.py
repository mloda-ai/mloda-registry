"""End-to-end worked example of mloda's ASOF (point-in-time) join.

This module is a *readable* example, not a red-phase failing test: mloda already
implements the ASOF join in core, so the test PASSES on first run. It is kept as
a regression/example that pins the point-in-time semantics, and the ASOF join
guide links to it. The same scenario is exercised on two compute frameworks,
pandas and PyArrow, via a single parametrized test; mloda 0.9.0 implements the
PyArrow ASOF join natively (Acero), so both backends pass.

Scenario (tiny and hand-traceable)
----------------------------------
Two source feature groups, both backed by ``DataCreator``:

* ``AsofEventSource`` (LEFT / events): columns ``id``, ``symbol``, ``event_ts``.
* ``AsofQuoteSource`` (RIGHT / quotes): columns ``symbol``, ``quote_ts``, ``price``.

Both declare ``index_columns() -> [Index(("symbol",))]`` so the ASOF link can
derive the equi by-key (``symbol``) automatically. ``Link.asof_on(...)`` then
performs a *backward* as-of join: for each event row it picks the quote with the
same ``symbol`` whose ``quote_ts`` is the latest value ``<= event_ts``.
``allow_exact_matches=True`` (the default) means an event whose timestamp equals
a quote timestamp matches that quote.

Quote price timelines (per symbol)::

    AAA:  10:00 -> 100      BBB:  10:00 -> 200
          10:30 -> 110            10:30 -> 220

Hand-traced expected price in effect at each event (backward, exact allowed)::

    id  symbol  event_ts   reasoning                                  -> price
    0   AAA     10:00      exact match on AAA@10:00                    -> 100
    1   AAA     10:15      latest AAA quote <= 10:15 is AAA@10:00      -> 100
    2   AAA     10:45      latest AAA quote <= 10:45 is AAA@10:30      -> 110
    3   BBB     10:30      exact match on BBB@10:30                    -> 220
    4   BBB     11:00      latest BBB quote <= 11:00 is BBB@10:30      -> 220

The join is keyed by ``symbol`` (the by-key), so AAA events never see BBB quotes
and vice versa. Every event sits at or after its symbol's first quote, so there
are no null matches in this example.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import pytest

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")

from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda.provider import ComputeFramework, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Index, Link, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

_U = timezone.utc

# Expected price in effect at each event id (see module docstring for the trace).
_ASOF_BACKWARD_EXPECTED: dict[int, int] = {0: 100, 1: 100, 2: 110, 3: 220, 4: 220}


def _to_frame(framework: type[ComputeFramework], columns: dict[str, Any]) -> Any:
    """Materialize a column dict into the native frame of ``framework``.

    Both source and consumer feature groups produce their output this way so the
    per-backend difference is confined to a single place: pandas builds a
    ``DataFrame`` and PyArrow builds a ``Table``. ``columns`` may hold plain
    Python lists (from the sources) or already-native columns such as a PyArrow
    ``ChunkedArray`` (from the merged frame in the consumer).
    """
    if framework is PandasDataFrame:
        return pd.DataFrame(columns)
    return pa.table(columns)


class _AsofEventSource(FeatureGroup):
    """LEFT side base: event rows carrying the entity (``symbol``) and event time.

    All backend-agnostic wiring lives here; concrete subclasses only pin the
    ``compute_framework`` so ``calculate_feature`` materializes the matching
    native frame via :func:`_to_frame`.
    """

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
    """RIGHT side base: a per-symbol price timeline keyed by quote time."""

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
    """Consumer base that pulls columns from both sources, triggering the ASOF link.

    Requesting features from two different feature groups that have a ``Link``
    between them is what makes mloda apply the join: the merged frame is handed
    to ``calculate_feature`` as ``data``. This group then exposes the event id
    and the as-of price as two named features so the test can assert keyed by id
    (the merge engine sorts the result by ``event_ts``, so a positional list
    would not be in input order). The output frame is materialized in the
    subclass's ``compute_framework`` so the result matches the requested backend.
    """

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


def _extract_asof_prices(results: list[Any]) -> dict[int, int]:
    """Read the ``{id: price}`` mapping from the pandas or PyArrow result frame."""
    for table in results:
        if isinstance(table, pd.DataFrame) and "asof_event_price" in table.columns:
            return {int(i): int(p) for i, p in zip(table["asof_event_id"], table["asof_event_price"])}
        if isinstance(table, pa.Table) and "asof_event_price" in table.column_names:
            ids = table.column("asof_event_id").to_pylist()
            prices = table.column("asof_event_price").to_pylist()
            return {int(i): int(p) for i, p in zip(ids, prices)}

    raise AssertionError("No result frame with asof_event_price found")


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

    return _extract_asof_prices(results)


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
