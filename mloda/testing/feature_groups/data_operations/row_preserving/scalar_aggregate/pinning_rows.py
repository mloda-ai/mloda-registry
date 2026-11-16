"""Pinning rows of scalar aggregate, written by hand from the rules in ``AGGREGATE_RULES``."""

from __future__ import annotations

from mloda.testing.feature_groups.data_operations.pinning import AGGREGATE_RULES, PinningRow, PinningTable

NAN = float("nan")

PINNING = PinningTable(
    operation="scalar_aggregate",
    rules=AGGREGATE_RULES,
    rows=(
        PinningRow("none", "v__sum_scalar", {"v": (1, None, 5)}, {"v__sum_scalar": (6, 6, 6)}),
        PinningRow("none", "v__count_scalar", {"v": (None, None)}, {"v__count_scalar": (0, 0)}, types={"v": "int64"}),
        PinningRow("none", "v__std_scalar", {"v": (None, None)}, {"v__std_scalar": (None, None)}, types={"v": "int64"}),
        PinningRow("nan", "v__avg_scalar", {"v": (1.0, NAN, 3.0)}, {"v__avg_scalar": (NAN, NAN, NAN)}),
        PinningRow("nan", "v__median_scalar", {"v": (NAN, 1.0, 3.0)}, {"v__median_scalar": (2.0, 2.0, 2.0)}),
        PinningRow("nan", "v__max_scalar", {"v": (NAN, NAN)}, {"v__max_scalar": (NAN, NAN)}),
        PinningRow("std_var", "v__var_samp_scalar", {"v": (2.0, 4.0)}, {"v__var_samp_scalar": (2.0, 2.0)}),
        PinningRow("std_var", "v__std_pop_scalar", {"v": (5.0,)}, {"v__std_pop_scalar": (0.0,)}),
        PinningRow("std_var", "v__var_samp_scalar", {"v": (5.0,)}, {"v__var_samp_scalar": (None,)}),
        PinningRow("row_order", "v__min_scalar", {"v": (3, 1, 2)}, {"v": (3, 1, 2), "v__min_scalar": (1, 1, 1)}),
        PinningRow(
            "dtype",
            "v__sum_scalar",
            {"v": (200, 100)},
            {"v__sum_scalar": (300, 300)},
            types={"v": "uint8", "v__sum_scalar": "uint64"},
        ),
        PinningRow(
            "dtype",
            "v__avg_scalar",
            {"v": (1, 2)},
            {"v__avg_scalar": (1.5, 1.5)},
            types={"v": "int32", "v__avg_scalar": "float64"},
        ),
    ),
)
