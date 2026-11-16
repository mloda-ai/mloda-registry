"""Pinning rows of aggregation, written by hand from the rules in ``AGGREGATE_RULES``."""

from __future__ import annotations

from mloda.testing.feature_groups.data_operations.pinning import AGGREGATE_RULES, PinningRow, PinningTable

NAN = float("nan")
BY_G = {"partition_by": ["g"]}

PINNING = PinningTable(
    operation="aggregation",
    rules=AGGREGATE_RULES,
    rows=(
        PinningRow(
            "none",
            "v__sum_agg",
            {"g": ("a", "a", "b"), "v": (1, None, 5)},
            {"g": ("a", "b"), "v__sum_agg": (1, 5)},
            BY_G,
        ),
        PinningRow("none", "v__count_agg", {"g": ("a", "a", "a"), "v": (1, None, 3)}, {"v__count_agg": (2,)}, BY_G),
        PinningRow(
            "none", "v__count_agg", {"g": ("a", "a", "b", "b"), "v": (None, None, 2, 3)}, {"v__count_agg": (0, 2)}, BY_G
        ),
        PinningRow(
            "none", "v__sum_agg", {"g": ("a", "a", "b", "b"), "v": (None, None, 2, 3)}, {"v__sum_agg": (None, 5)}, BY_G
        ),
        PinningRow("nan", "v__sum_agg", {"g": ("a", "a", "b"), "v": (1.0, NAN, 2.0)}, {"v__sum_agg": (NAN, 2.0)}, BY_G),
        PinningRow("nan", "v__count_agg", {"g": ("a", "a"), "v": (NAN, 1.0)}, {"v__count_agg": (2,)}, BY_G),
        PinningRow("nan", "v__min_agg", {"g": ("a", "a", "b"), "v": (NAN, 1.0, NAN)}, {"v__min_agg": (1.0, NAN)}, BY_G),
        PinningRow("nan", "v__max_agg", {"g": ("a", "a", "b"), "v": (NAN, 1.0, NAN)}, {"v__max_agg": (1.0, NAN)}, BY_G),
        PinningRow(
            "nan",
            "v__median_agg",
            {"g": ("a", "a", "a", "b"), "v": (NAN, 1.0, 3.0, NAN)},
            {"v__median_agg": (2.0, None)},
            BY_G,
        ),
        PinningRow(
            "nan", "v__first_agg", {"g": ("a", "a", "a"), "v": (None, NAN, 1.0)}, {"v__first_agg": (NAN,)}, BY_G
        ),
        PinningRow(
            "nan",
            "v__nunique_agg",
            {"g": ("a", "a", "a", "a"), "v": (NAN, NAN, 1.0, None)},
            {"v__nunique_agg": (2,)},
            BY_G,
        ),
        PinningRow(
            "signed_zero",
            "v__nunique_agg",
            {"g": ("a", "a", "a"), "v": (0.0, -0.0, -0.0)},
            {"v__nunique_agg": (1,)},
            BY_G,
        ),
        PinningRow(
            "signed_zero", "v__sum_agg", {"g": (0.0, -0.0), "v": (1, 2)}, {"g": (0.0,), "v__sum_agg": (3,)}, BY_G
        ),
        PinningRow(
            "mode",
            "v__mode_agg",
            {"g": ("a",) * 5, "v": (2, 1, 1, 2, 3)},
            {"v__mode_agg": (2,)},
            BY_G,
            {"v__mode_agg": "int64"},
        ),
        PinningRow(
            "std_var",
            "v__std_pop_agg",
            {"g": ("a", "a", "b"), "v": (1.0, 3.0, 5.0)},
            {"v__std_pop_agg": (1.0, 0.0)},
            BY_G,
        ),
        PinningRow(
            "std_var",
            "v__var_samp_agg",
            {"g": ("a", "a", "b"), "v": (1.0, 3.0, 5.0)},
            {"v__var_samp_agg": (2.0, None)},
            BY_G,
        ),
        PinningRow(
            "std_var",
            "v__std_samp_agg",
            {"g": ("a", "a", "b"), "v": (1.0, 3.0, 5.0)},
            {"v__std_samp_agg": (1.4142135623730951, None)},
            BY_G,
        ),
        PinningRow(
            "partition_keys",
            "v__sum_agg",
            {"g": (NAN, None, 1.0, NAN, None), "v": (1, 2, 3, 4, 5)},
            {"g": (NAN, None, 1.0), "v__sum_agg": (5, 7, 3)},
            BY_G,
        ),
        PinningRow(
            "partition_keys", "v__count_agg", {"v": ()}, {"v__count_agg": (0,)}, {"partition_by": []}, {"v": "int64"}
        ),
        PinningRow(
            "dtype",
            "v__sum_agg",
            {"g": ("a", "a"), "v": (1, 2)},
            {"v__sum_agg": (3,)},
            BY_G,
            {"v": "int32", "v__sum_agg": "int64"},
        ),
        PinningRow(
            "dtype",
            "v__sum_agg",
            {"g": ("a", "a"), "v": (200, 100)},
            {"v__sum_agg": (300,)},
            BY_G,
            {"v": "uint8", "v__sum_agg": "uint64"},
        ),
        PinningRow(
            "dtype",
            "v__count_agg",
            {"g": ("a",), "v": (1,)},
            {"v__count_agg": (1,)},
            BY_G,
            {"v": "int32", "v__count_agg": "int64"},
        ),
        PinningRow(
            "dtype",
            "v__avg_agg",
            {"g": ("a", "a"), "v": (1, 2)},
            {"v__avg_agg": (1.5,)},
            BY_G,
            {"v": "int32", "v__avg_agg": "float64"},
        ),
        PinningRow(
            "dtype",
            "v__min_agg",
            {"g": ("a", "a"), "v": (4, 2)},
            {"v__min_agg": (2,)},
            BY_G,
            {"v": "int32", "v__min_agg": "int32"},
        ),
        PinningRow(
            "dtype",
            "v__sum_agg",
            {"g": ("a", "a", "a"), "v": (True, False, True)},
            {"v__sum_agg": (2,)},
            BY_G,
            {"v": "bool", "v__sum_agg": "uint64"},
        ),
        PinningRow("nan", "v__last_agg", {"g": ("a", "a", "a"), "v": (1.0, NAN, None)}, {"v__last_agg": (NAN,)}, BY_G),
        PinningRow("nan", "v__std_agg", {"g": ("a", "a"), "v": (1.0, NAN)}, {"v__std_agg": (NAN,)}, BY_G),
        PinningRow(
            "nan",
            "v__mode_agg",
            {"g": ("a",) * 4, "v": (NAN, 1.0, NAN, 1.0)},
            {"v__mode_agg": (NAN,)},
            BY_G,
        ),
    ),
)
