"""Hand-computed pinning rows of the sessionization reference."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from types import MappingProxyType

from mloda.user import DataType

from mloda.testing.feature_groups.data_operations.pinning import PinningRow, PinningTable

NAN = float("nan")

SESSIONIZATION_RULES: Mapping[str, str] = MappingProxyType(
    {
        "order": "Rows are sorted by [*partition_by, order_by] ascending; the gap is measured on order_by.",
        "ties": "Rows tied on order_by have gap zero and share a session.",
        "gap": "A row starts a new session when its gap to the previous row is greater than the threshold; an equal gap stays.",
        "none": "A None order value starts its own session; None sorts last within its partition.",
        "partition_keys": "A partition's first row starts a session; session ids are one global counter over the sorted frame, "
        "partitions ordered by key value (real values, then NaN, then None).",
        "nan_none_keys": "NaN partition keys merge into one partition; a None partition key is its own partition.",
        "whole_table": "Without partition_by the whole table is one partition.",
        "dtype": "The session id is int64, whatever the timestamp resolution.",
        "empty": "An empty input gives an empty int64 column.",
    }
)

_I = DataType.INT64
_S = DataType.STRING
_D = DataType.DOUBLE
_TS = DataType.TIMESTAMP_MICROS_TZ
_CTX = {"partition_by": ["p"], "order_by": "ts"}
_M30 = "ts__sessionize_30_minute"


def _t(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2023, 1, 1, hour, minute, second, tzinfo=timezone.utc)


SESSIONIZATION_PINNING = PinningTable(
    operation="sessionization",
    rules=SESSIONIZATION_RULES,
    rows=(
        PinningRow(
            case="gap",
            feature=_M30,
            inputs={"p": (_S, ("a",) * 4), "ts": (_TS, (_t(10), _t(10, 30), _t(11, 0, 1), _t(11, 10)))},
            # gaps 30m (equal, stays), 30m1s (new), 9m59s
            expected={_M30: (_I, (0, 0, 1, 1))},
            context=_CTX,
        ),
        PinningRow(
            case="gap",
            feature="ts__sessionize_1_hour",
            inputs={"p": (_S, ("a",) * 3), "ts": (_TS, (_t(10), _t(11), _t(12, 0, 1)))},
            expected={"ts__sessionize_1_hour": (_I, (0, 0, 1))},
            context=_CTX,
        ),
        PinningRow(
            case="order",
            feature=_M30,
            inputs={"p": (_S, ("a",) * 3), "ts": (_TS, (_t(12), _t(10), _t(10, 10)))},
            # order idx 1, 2, 0 holds 10:00, 10:10, 12:00
            expected={_M30: (_I, (1, 0, 0))},
            context=_CTX,
        ),
        PinningRow(
            case="order",
            feature=_M30,
            inputs={
                "p": (_S, ("a", "a")),
                "ts": (_TS, (_t(10), _t(10))),
                "o": (_TS, (_t(10), _t(12))),
            },
            # the gap is measured on order_by (2h), not on the source column
            expected={_M30: (_I, (0, 1))},
            context={"partition_by": ["p"], "order_by": "o"},
        ),
        PinningRow(
            case="ties",
            feature=_M30,
            inputs={"p": (_S, ("a",) * 3), "ts": (_TS, (_t(10), _t(10), _t(10)))},
            expected={_M30: (_I, (0, 0, 0))},
            context=_CTX,
        ),
        PinningRow(
            case="none",
            feature=_M30,
            inputs={"p": (_S, ("a",) * 3), "ts": (_TS, (None, _t(10), None))},
            # order idx 1, 0, 2: the value first, then each None on its own, in input order
            expected={_M30: (_I, (1, 0, 2))},
            context=_CTX,
        ),
        PinningRow(
            case="none",
            feature=_M30,
            inputs={"p": (_S, ("a", "a", "b")), "ts": (_TS, (None, None, None))},
            expected={_M30: (_I, (0, 1, 2))},
            context=_CTX,
        ),
        PinningRow(
            case="partition_keys",
            feature=_M30,
            inputs={"p": (_S, ("b", "a", "b", "a")), "ts": (_TS, (_t(10), _t(10), _t(10, 10), _t(12)))},
            # sorted frame: a 10:00 (0), a 12:00 (1), b 10:00 (2), b 10:10 (2)
            expected={_M30: (_I, (2, 0, 2, 1))},
            context=_CTX,
        ),
        PinningRow(
            case="nan_none_keys",
            feature=_M30,
            inputs={
                "p": (_D, (None, NAN, 1.0, NAN, None)),
                "ts": (_TS, (_t(10), _t(10), _t(10), _t(10, 10), _t(10, 20))),
            },
            # sorted frame: 1.0 idx2 (0), NaN idx1 and idx3 (1, merged), None idx0 and idx4 (2)
            expected={_M30: (_I, (2, 1, 0, 1, 2))},
            context=_CTX,
        ),
        PinningRow(
            case="whole_table",
            feature=_M30,
            inputs={"ts": (_TS, (_t(10), _t(12), _t(10, 10)))},
            # order idx 0, 2, 1
            expected={_M30: (_I, (0, 1, 0))},
            context={"order_by": "ts"},
        ),
        PinningRow(
            case="dtype",
            feature="ts__sessionize_1_minute",
            inputs={
                "p": (_S, ("a",) * 3),
                "ts": (
                    DataType.TIMESTAMP_MILLIS,
                    (datetime(2023, 1, 1, 10), datetime(2023, 1, 1, 10, 0, 30), datetime(2023, 1, 1, 10, 2)),
                ),
            },
            # gaps 30s (stays), 90s (new)
            expected={"ts__sessionize_1_minute": (_I, (0, 0, 1))},
            context=_CTX,
        ),
        PinningRow(
            case="empty",
            feature=_M30,
            inputs={"p": (_S, ()), "ts": (_TS, ())},
            expected={_M30: (_I, ())},
            context=_CTX,
        ),
    ),
)
