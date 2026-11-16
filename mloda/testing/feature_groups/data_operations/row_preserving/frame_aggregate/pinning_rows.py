"""Pinning rows of frame aggregate, written by hand from the rules in ``AGGREGATE_RULES``."""

from __future__ import annotations

from mloda.testing.feature_groups.data_operations.pinning import AGGREGATE_RULES, PinningRow, PinningTable

NAN = float("nan")
BY_REGION_TS = {"partition_by": ["region"], "order_by": "ts"}
A4 = ("A",) * 4

PINNING = PinningTable(
    operation="frame_aggregate",
    rules=AGGREGATE_RULES,
    rows=(
        PinningRow(
            "none",
            "value__sum_rolling_3",
            {"region": A4, "ts": (1, 2, 3, 4), "value": (10, None, 30, 40)},
            {"value__sum_rolling_3": (10, 10, 40, 70)},
            BY_REGION_TS,
        ),
        PinningRow(
            "none",
            "value__cumsum",
            {"region": ("A", "A"), "ts": (1, 2), "value": (None, None)},
            {"value__cumsum": (None, None)},
            BY_REGION_TS,
        ),
        PinningRow(
            "none",
            "value__cumcount",
            {"region": ("A", "A"), "ts": (1, 2), "value": (None, None)},
            {"value__cumcount": (0, 0)},
            BY_REGION_TS,
        ),
        PinningRow(
            "none",
            "value__sum_rolling_2",
            {"region": ("A",) * 3, "ts": (1, 2, 3), "value": (None, None, 5)},
            {"value__sum_rolling_2": (None, None, 5)},
            BY_REGION_TS,
        ),
        PinningRow(
            "nan",
            "value__avg_rolling_2",
            {"region": A4, "ts": (1, 2, 3, 4), "value": (1.0, NAN, 3.0, 5.0)},
            {"value__avg_rolling_2": (1.0, NAN, NAN, 4.0)},
            BY_REGION_TS,
        ),
        PinningRow(
            "nan",
            "value__max_rolling_2",
            {"region": A4, "ts": (1, 2, 3, 4), "value": (1.0, NAN, 3.0, 5.0)},
            {"value__max_rolling_2": (1.0, 1.0, 3.0, 5.0)},
            BY_REGION_TS,
        ),
        PinningRow(
            "std_var",
            "value__std_rolling_2",
            {"region": ("A",) * 3, "ts": (1, 2, 3), "value": (1.0, 3.0, 3.0)},
            {"value__std_rolling_2": (0.0, 1.0, 0.0)},
            BY_REGION_TS,
        ),
        PinningRow(
            "std_var",
            "value__var_rolling_2",
            {"region": ("A",) * 3, "ts": (1, 2, 3), "value": (1.0, 3.0, 3.0)},
            {"value__var_rolling_2": (0.0, 1.0, 0.0)},
            BY_REGION_TS,
        ),
        PinningRow(
            "order",
            "value__cumsum",
            {"region": ("A",) * 3, "ts": (None, 1, 2), "value": (100, 10, 20)},
            {"value__cumsum": (130, 10, 30)},
            BY_REGION_TS,
        ),
        PinningRow(
            "order",
            "value__cumsum",
            {"region": ("A",) * 4, "ts": (None, 3.0, NAN, 1.0), "value": (100, 4, 8, 1)},
            {"value__cumsum": (113, 5, 13, 1)},
            BY_REGION_TS,
        ),
        PinningRow(
            "row_order",
            "value__cumsum",
            {"region": A4, "ts": (3.0, 1.0, NAN, 2.0), "value": (4, 1, 8, 2)},
            {"value__cumsum": (7, 1, 15, 3)},
            BY_REGION_TS,
        ),
        PinningRow(
            "partition_keys",
            "value__cumsum",
            {"region": (NAN, NAN, None, 1.0, None), "ts": (1, 2, 3, 4, 5), "value": (1, 2, 3, 4, 5)},
            {"value__cumsum": (1, 3, 3, 4, 8)},
            BY_REGION_TS,
        ),
        PinningRow(
            "signed_zero",
            "value__cumsum",
            {"region": (0.0, -0.0, 0.0), "ts": (1, 2, 3), "value": (1, 2, 4)},
            {"value__cumsum": (1, 3, 7)},
            BY_REGION_TS,
        ),
        PinningRow(
            "dtype",
            "value__cumsum",
            {"region": ("A", "A"), "ts": (1, 2), "value": (1, 2)},
            {"value__cumsum": (1, 3)},
            BY_REGION_TS,
            {"value": "int32", "value__cumsum": "int64"},
        ),
        PinningRow(
            "dtype",
            "value__cumavg",
            {"region": ("A", "A"), "ts": (1, 2), "value": (1, 2)},
            {"value__cumavg": (1.0, 1.5)},
            BY_REGION_TS,
            {"value": "int32", "value__cumavg": "float64"},
        ),
    ),
)
