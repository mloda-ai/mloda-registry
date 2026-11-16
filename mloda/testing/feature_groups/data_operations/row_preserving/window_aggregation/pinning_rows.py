"""Pinning rows of window aggregation, written by hand from the rules in ``AGGREGATE_RULES``."""

from __future__ import annotations

from mloda.testing.feature_groups.data_operations.pinning import AGGREGATE_RULES, PinningRow, PinningTable

NAN = float("nan")
BY_G = {"partition_by": ["g"]}
BY_G_O = {"partition_by": ["g"], "order_by": "o"}

PINNING = PinningTable(
    operation="window_aggregation",
    rules=AGGREGATE_RULES,
    rows=(
        PinningRow(
            "none",
            "v__sum_window",
            {"g": ("a", "b", "a", "b", "a"), "v": (1, None, 2, 4, None)},
            {"v__sum_window": (3, 4, 3, 4, 3)},
            BY_G,
        ),
        PinningRow(
            "none",
            "v__count_window",
            {"g": ("a", "a", "b"), "v": (None, None, 7)},
            {"v__count_window": (0, 0, 1)},
            BY_G,
        ),
        PinningRow(
            "none",
            "v__sum_window",
            {"g": ("a", "a", "b"), "v": (None, None, 7)},
            {"v__sum_window": (None, None, 7)},
            BY_G,
        ),
        PinningRow(
            "none",
            "v__first_window",
            {"g": ("a", "a"), "v": (None, 7), "o": (1, 2)},
            {"v__first_window": (7, 7)},
            BY_G_O,
        ),
        PinningRow(
            "nan",
            "v__avg_window",
            {"g": ("a", "a", "b"), "v": (1.0, NAN, 2.0)},
            {"v__avg_window": (NAN, NAN, 2.0)},
            BY_G,
        ),
        PinningRow(
            "nan",
            "v__median_window",
            {"g": ("a", "a", "a"), "v": (NAN, 1.0, 3.0)},
            {"v__median_window": (2.0, 2.0, 2.0)},
            BY_G,
        ),
        PinningRow(
            "order",
            "v__first_window",
            {"g": ("a",) * 4, "v": (10, 20, 30, 40), "o": (3.0, NAN, None, 1.0)},
            {"v__first_window": (40, 40, 40, 40)},
            BY_G_O,
        ),
        PinningRow(
            "order",
            "v__last_window",
            {"g": ("a",) * 4, "v": (10, 20, 30, 40), "o": (3.0, NAN, None, 1.0)},
            {"v__last_window": (30, 30, 30, 30)},
            BY_G_O,
        ),
        PinningRow(
            "order",
            "v__first_window",
            {"g": ("a",) * 3, "v": (1, 2, 3), "o": (5, 5, 5)},
            {"v__first_window": (1, 1, 1)},
            BY_G_O,
        ),
        PinningRow(
            "order",
            "v__last_window",
            {"g": ("a",) * 3, "v": (1, 2, 3), "o": (5, 5, 5)},
            {"v__last_window": (3, 3, 3)},
            BY_G_O,
        ),
        PinningRow(
            "row_order",
            "v__min_window",
            {"g": ("b", "a", "b", "a"), "v": (1, 10, 2, 20)},
            {"v__min_window": (1, 10, 1, 10)},
            BY_G,
        ),
        PinningRow(
            "dtype",
            "v__sum_window",
            {"g": ("a", "a"), "v": (1, 2)},
            {"v__sum_window": (3, 3)},
            BY_G,
            {"v": "int32", "v__sum_window": "int64"},
        ),
    ),
)
