"""Hand-computed pinning rows of the EMA reference.

The four fixture lists (``EXPECTED_EMA_*``) are the single home of the 12-row fixture values, computed
offline from the canonical pandas formula per partition, in ORIGINAL ROW ORDER::

    for region, group in df.groupby("region"):
        s = group.sort_values("ts")["value"]
        ema = s.ewm(span=SPAN, adjust=False, ignore_na=True).mean().mask(s.isna())

The whole-table variant sorts the entire frame by ts (no groupby). Polars span-form
``ewm_mean(span=SPAN, adjust=False, ignore_nulls=True)`` agrees elementwise (null-aware, within 1e-9).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

from mloda.user import DataType

from mloda.testing.feature_groups.data_operations.pinning import PinningRow, PinningTable

NAN = float("nan")

EMA_RULES: Mapping[str, str] = MappingProxyType(
    {
        "fixture": "The 12-row fixture gives the canonical pandas ewm values (span 2 and 3, partitioned and whole table).",
        "recurrence": "alpha = 2 / (span + 1), adjust=False: ema = alpha * x + (1 - alpha) * previous ema; the first non-None seeds.",
        "order": "Each partition is ordered by order_by ascending, stably: input order breaks ties.",
        "order_tiers": "In order_by real values come first, then NaN, then None; NaN and None are separate tiers.",
        "none": "A None input gives None and does not advance the state; an all-None group stays None.",
        "nan": "NaN in the source is a value: it propagates into the state, so its row and every later non-None row are NaN.",
        "partition_keys": "NaN partition keys merge into one group; a None partition key is its own group.",
        "whole_table": "Without partition_by the whole table is one group.",
        "dtype": "The output is float64, also for an integer source.",
        "empty": "An empty input gives an empty float64 column.",
    }
)

_I = DataType.INT64
_D = DataType.DOUBLE
_S = DataType.STRING
_CTX = {"partition_by": ["p"], "order_by": "o"}

# The 12-row fixture of ``EmaTestBase`` (id is its row-order witness): two interleaved regions, unique timestamps.
EMA_IDS: list[int] = list(range(12))
EMA_REGIONS: list[str] = ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]
# Hour-of-day per row (all globally unique, so SQL whole-table sorts are deterministic).
EMA_TIMESTAMPS: list[datetime] = [
    datetime(2023, 1, 1, h, tzinfo=timezone.utc) for h in (2, 7, 0, 9, 4, 6, 1, 11, 5, 8, 3, 10)
]
EMA_VALUES: list[float | None] = [2.0, 10.0, None, None, 4.0, None, 1.0, None, None, 20.0, None, 40.0]
_FIXTURE = {
    "region": (_S, tuple(EMA_REGIONS)),
    "ts": (DataType.TIMESTAMP_MICROS_TZ, tuple(EMA_TIMESTAMPS)),
    "value": (_D, tuple(EMA_VALUES)),
}
_FIXTURE_PART = {"partition_by": ["region"], "order_by": "ts"}

# Per-partition EMA, span=2 (partition_by=["region"], order_by="ts").
EXPECTED_EMA_SPAN2: list[Any] = [
    1.6666666666666665,  # 0  A
    10.0,  # 1  B seed
    None,  # 2  A leading null
    None,  # 3  B interior null
    3.2222222222222223,  # 4  A
    None,  # 5  B leading null
    1.0,  # 6  A seed
    None,  # 7  B trailing null
    None,  # 8  A trailing null
    16.666666666666664,  # 9  B
    None,  # 10 A interior null
    32.22222222222222,  # 11 B
]

# Per-partition EMA, span=3.
EXPECTED_EMA_SPAN3: list[Any] = [
    1.5,  # 0  A
    10.0,  # 1  B seed
    None,  # 2  A leading null
    None,  # 3  B interior null
    2.75,  # 4  A
    None,  # 5  B leading null
    1.0,  # 6  A seed
    None,  # 7  B trailing null
    None,  # 8  A trailing null
    15.0,  # 9  B
    None,  # 10 A interior null
    27.5,  # 11 B
]

# Whole-table EMA, span=2 (order_by="ts" only, no partition). Differs from the
# per-partition result at rows 1, 9, 11 (partition bleed guard).
EXPECTED_EMA_WHOLE_SPAN2: list[Any] = [
    1.6666666666666665,  # 0
    7.7407407407407405,  # 1
    None,  # 2
    None,  # 3
    3.2222222222222223,  # 4
    None,  # 5
    1.0,  # 6
    None,  # 7
    None,  # 8
    15.913580246913579,  # 9
    None,  # 10
    31.971193415637856,  # 11
]

# Whole-table EMA, span=3.
EXPECTED_EMA_WHOLE_SPAN3: list[Any] = [
    1.5,  # 0
    6.375,  # 1
    None,  # 2
    None,  # 3
    2.75,  # 4
    None,  # 5
    1.0,  # 6
    None,  # 7
    None,  # 8
    13.1875,  # 9
    None,  # 10
    26.59375,  # 11
]

EMA_PINNING = PinningTable(
    operation="ema",
    rules=EMA_RULES,
    rows=(
        PinningRow(
            case="fixture",
            feature="value__ema_2",
            inputs=_FIXTURE,
            expected={"value__ema_2": (_D, tuple(EXPECTED_EMA_SPAN2))},
            context=_FIXTURE_PART,
        ),
        PinningRow(
            case="fixture",
            feature="value__ema_3",
            inputs=_FIXTURE,
            expected={"value__ema_3": (_D, tuple(EXPECTED_EMA_SPAN3))},
            context=_FIXTURE_PART,
        ),
        PinningRow(
            case="fixture",
            feature="value__ema_2",
            inputs=_FIXTURE,
            expected={"value__ema_2": (_D, tuple(EXPECTED_EMA_WHOLE_SPAN2))},
            context={"order_by": "ts"},
        ),
        PinningRow(
            case="fixture",
            feature="value__ema_3",
            inputs=_FIXTURE,
            expected={"value__ema_3": (_D, tuple(EXPECTED_EMA_WHOLE_SPAN3))},
            context={"order_by": "ts"},
        ),
        PinningRow(
            case="recurrence",
            feature="v__ema_3",
            inputs={"p": (_S, ("a",) * 4), "o": (_I, (1, 2, 3, 4)), "v": (_D, (4.0, 8.0, None, 2.0))},
            # alpha 0.5: 4, 6, None (state stays 6), 0.5 * 2 + 0.5 * 6
            expected={"v__ema_3": (_D, (4.0, 6.0, None, 4.0))},
            context=_CTX,
        ),
        PinningRow(
            case="recurrence",
            feature="v__ema_1",
            inputs={"p": (_S, ("a",) * 3), "o": (_I, (1, 2, 3)), "v": (_D, (3.0, 7.0, 5.0))},
            # span 1 gives alpha 1: each output is its input
            expected={"v__ema_1": (_D, (3.0, 7.0, 5.0))},
            context=_CTX,
        ),
        PinningRow(
            case="order",
            feature="v__ema_3",
            inputs={"p": (_S, ("a",) * 3), "o": (_I, (3, 1, 2)), "v": (_D, (4.0, 8.0, 2.0))},
            # order idx 1, 2, 0 holds 8, 2, 4: ema 8, 5, 4.5
            expected={"v__ema_3": (_D, (4.5, 8.0, 5.0))},
            context=_CTX,
        ),
        PinningRow(
            case="order",
            feature="v__ema_3",
            inputs={"p": (_S, ("a",) * 3), "o": (_I, (1, 1, 1)), "v": (_D, (4.0, 8.0, 2.0))},
            # tied: input order 4, 8, 2 gives 4, 6, 4
            expected={"v__ema_3": (_D, (4.0, 6.0, 4.0))},
            context=_CTX,
        ),
        PinningRow(
            case="order_tiers",
            feature="v__ema_3",
            inputs={"p": (_S, ("a",) * 3), "o": (_D, (None, NAN, 1.0)), "v": (_D, (2.0, 6.0, 4.0))},
            # order idx 2, 1, 0 holds 4, 6, 2: ema 4, 5, 3.5
            expected={"v__ema_3": (_D, (3.5, 5.0, 4.0))},
            context=_CTX,
        ),
        PinningRow(
            case="order_tiers",
            feature="v__ema_3",
            inputs={"p": (_S, ("a",) * 3), "o": (_I, (None, 1, 2)), "v": (_D, (2.0, 4.0, 8.0))},
            # None order value sorts last: order idx 1, 2, 0 holds 4, 8, 2: ema 4, 6, 4
            expected={"v__ema_3": (_D, (4.0, 4.0, 6.0))},
            context=_CTX,
        ),
        PinningRow(
            case="none",
            feature="v__ema_3",
            inputs={"p": (_S, ("a", "a", "b")), "o": (_I, (1, 2, 1)), "v": (_D, (None, None, 8.0))},
            expected={"v__ema_3": (_D, (None, None, 8.0))},
            context=_CTX,
        ),
        PinningRow(
            case="none",
            feature="v__ema_3",
            inputs={"p": (_S, ("a",) * 3), "o": (_I, (1, 2, 3)), "v": (_D, (None, 4.0, None))},
            expected={"v__ema_3": (_D, (None, 4.0, None))},
            context=_CTX,
        ),
        PinningRow(
            case="nan",
            feature="v__ema_2",
            inputs={"p": (_S, ("a",) * 5), "o": (_I, (1, 2, 3, 4, 5)), "v": (_D, (10.0, NAN, 30.0, None, 50.0))},
            # seed 10, NaN poisons the state: NaN, NaN, None stays None, NaN
            expected={"v__ema_2": (_D, (10.0, NAN, NAN, None, NAN))},
            context=_CTX,
        ),
        PinningRow(
            case="partition_keys",
            feature="v__ema_3",
            inputs={"p": (_D, (NAN, 1.0, NAN)), "o": (_I, (1, 1, 2)), "v": (_D, (4.0, 10.0, 8.0))},
            expected={"v__ema_3": (_D, (4.0, 10.0, 6.0))},
            context=_CTX,
        ),
        PinningRow(
            case="partition_keys",
            feature="v__ema_3",
            inputs={"p": (_D, (None, NAN, None, NAN)), "o": (_I, (1, 1, 2, 2)), "v": (_D, (2.0, 4.0, 6.0, 8.0))},
            expected={"v__ema_3": (_D, (2.0, 4.0, 4.0, 6.0))},
            context=_CTX,
        ),
        PinningRow(
            case="whole_table",
            feature="v__ema_3",
            inputs={"o": (_I, (2, 1, 3)), "v": (_D, (8.0, 4.0, 2.0))},
            # order idx 1, 0, 2 holds 4, 8, 2: ema 4, 6, 4
            expected={"v__ema_3": (_D, (6.0, 4.0, 4.0))},
            context={"order_by": "o"},
        ),
        PinningRow(
            case="dtype",
            feature="v__ema_3",
            inputs={"p": (_S, ("a",) * 2), "o": (_I, (1, 2)), "v": (_I, (2, 4))},
            expected={"v__ema_3": (_D, (2.0, 3.0))},
            context=_CTX,
        ),
        PinningRow(
            case="empty",
            feature="v__ema_3",
            inputs={"p": (_S, ()), "o": (_I, ()), "v": (_D, ())},
            expected={"v__ema_3": (_D, ())},
            context=_CTX,
        ),
    ),
)
