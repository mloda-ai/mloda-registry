"""Reproduced backend pinning limitations, keyed to exact rows rather than whole operations.

Every listed row still executes under strict xfail with the named exception, so a fixed backend
produces XPASS and the entry must shrink. A variant entry also requires the Implementation's
applicable declaration. Operations with their own twin pinned runner are not covered here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PinningLimitation:
    operation: str
    frameworks: tuple[str, ...]
    rows: tuple[str, ...]
    reason: str
    variant: str | None = None
    raises: str = "builtins.AssertionError"


LIMITATIONS: tuple[PinningLimitation, ...] = (
    PinningLimitation(
        "aggregation",
        ("pyarrow",),
        ("6-signed_zero-v__count_agg", "7-signed_zero-v__nunique_agg"),
        "Arrow hash aggregation distinguishes -0.0 and +0.0 as partition keys and unique values.",
    ),
    PinningLimitation(
        "aggregation",
        ("pandas",),
        ("0-nan-v__nunique_agg", "1-nan-val__first_agg"),
        "Pandas excludes NaN from nunique and skips a leading NaN in first.",
        variant="nan_as_null",
    ),
    PinningLimitation(
        "aggregation",
        ("pandas", "duckdb"),
        ("8-partition_keys-val__sum_agg",),
        "The NaN partition is returned after ordinary keys instead of in first-occurrence order.",
    ),
    PinningLimitation(
        "aggregation",
        ("sqlite",),
        ("6-signed_zero-v__count_agg",),
        "SQLite loses the negative sign of the first-seen zero partition key.",
    ),
    PinningLimitation(
        "aggregation",
        ("sqlite",),
        ("8-partition_keys-val__sum_agg",),
        "sqlite3 stores the NaN partition key as NULL.",
    ),
    PinningLimitation(
        "window_aggregation",
        ("pyarrow",),
        ("4-signed_zero-v__count_window", "5-signed_zero-v__nunique_window"),
        "Arrow hash grouping distinguishes the two zero signs for partitions and nunique.",
    ),
    PinningLimitation(
        "window_aggregation",
        ("pandas",),
        ("3-nan-v__first_window",),
        "Pandas first skips the leading NaN.",
        variant="nan_as_null",
    ),
    PinningLimitation(
        "window_aggregation",
        ("pandas",),
        ("6-order-v__last_window", "7-order-v__first_window"),
        "NaN and None tie in pandas order_by instead of forming distinct tiers.",
        variant="nan_as_null",
    ),
    PinningLimitation(
        "frame_aggregate",
        ("pandas", "sqlite"),
        ("8-order-v__cumsum",),
        "NaN and None tie in order_by, changing the cumulative sums.",
        variant="nan_as_null",
    ),
    PinningLimitation(
        "rank",
        ("pandas", "sqlite", "python_dict"),
        ("4-order_tiers-o__rank_ranked", "5-order_tiers-o__dense_rank_ranked"),
        "NaN and None tie in rank and dense_rank.",
        variant="nan_as_null",
    ),
    PinningLimitation(
        "rank",
        ("pandas", "sqlite", "python_dict"),
        ("3-order_tiers-o__row_number_ranked", "17-top_bottom-o__top_2_ranked"),
        "NaN and None also tie in row_number and top_N, outside the declared rank subtypes.",
    ),
    PinningLimitation(
        "rank",
        ("pandas", "sqlite"),
        ("20-partition_keys-o__row_number_ranked",),
        "NaN and None partition keys merge, so row_number crosses the reference partition boundary.",
    ),
    PinningLimitation(
        "rank",
        ("polars_lazy", "duckdb"),
        ("13-top_bottom-o__top_2_ranked",),
        "top_N selects the earlier tied maximum; the pinned rule reverses the stable ascending order.",
    ),
    PinningLimitation(
        "offset",
        ("pandas", "sqlite"),
        ("3-order_tiers-v__lag_1_offset", "4-order_tiers-v__lead_1_offset"),
        "NaN and None tie in order_by, changing the adjacent rows.",
        variant="nan_as_null",
    ),
    PinningLimitation(
        "offset",
        ("pandas", "sqlite"),
        (
            "7-nan-v__lag_1_offset",
            "8-nan-v__diff_1_offset",
            "14-nan-v__first_value_offset",
            "15-nan-v__last_value_offset",
        ),
        "NaN source values are treated as null by offset operations.",
        variant="nan_as_null",
    ),
    PinningLimitation(
        "offset",
        ("pandas", "sqlite"),
        ("17-partition_keys-v__lag_1_offset",),
        "NaN and None partition keys merge, so lag crosses the reference partition boundary.",
    ),
    PinningLimitation(
        "offset",
        ("polars_lazy",),
        ("20-dtype-v__diff_1_offset", "21-dtype-v__diff_1_offset"),
        "Narrow integer subtraction wraps before widening: UINT8 gives 252 instead of -4; INT8 gives -56 instead of 200.",
    ),
    PinningLimitation(
        "offset",
        ("duckdb",),
        ("20-dtype-v__diff_1_offset", "21-dtype-v__diff_1_offset"),
        "Narrow integer subtraction raises on UINT8 1 - 5 and INT8 100 - -100 before widening.",
        raises="duckdb.OutOfRangeException",
    ),
    PinningLimitation(
        "percentile",
        ("pandas",),
        (
            "4-endpoints-v__p0_percentile",
            "5-endpoints-v__p100_percentile",
            "6-endpoints-v__p50_percentile",
            "7-endpoints-v__p50_percentile",
        ),
        "Interpolation involving infinities returns null instead of the pinned endpoint or exact median.",
    ),
    PinningLimitation(
        "percentile",
        ("python_dict",),
        ("7-endpoints-v__p50_percentile",),
        "Interpolating repeated +inf values computes inf - inf and returns NaN instead of +inf.",
    ),
    PinningLimitation(
        "percentile",
        ("pandas",),
        ("16-partition_keys-v__p50_percentile",),
        "NaN and None partition keys merge, yielding a shared median of 6.5 instead of 2 and 15.",
    ),
    PinningLimitation(
        "ffill",
        ("pandas", "sqlite"),
        ("3-order_tiers-v__ffill",),
        "NaN and None tie in order_by, so ffill uses a different predecessor.",
        variant="nan_as_null",
    ),
    PinningLimitation(
        "ffill",
        ("pandas", "sqlite"),
        ("7-nan-v__ffill", "8-nan-v__ffill"),
        "A NaN source value is treated as a null to fill instead of as a value to carry.",
        variant="nan_as_null",
    ),
    PinningLimitation(
        "ffill",
        ("pandas", "sqlite"),
        ("10-partition_keys-v__ffill",),
        "NaN and None partition keys merge, carrying a value across the reference partition boundary.",
    ),
    PinningLimitation(
        "sessionization",
        ("pandas", "sqlite"),
        ("8-nan_none_keys-ts__sessionize_30_minute",),
        "NaN and None partition keys merge, changing session boundaries and IDs.",
    ),
    PinningLimitation(
        "ema",
        ("pandas",),
        ("8-order_tiers-v__ema_3",),
        "NaN and None tie in order_by, changing the EMA recurrence order.",
        variant="nan_as_null",
    ),
    PinningLimitation(
        "ema",
        ("pandas",),
        ("12-nan-v__ema_2",),
        "Pandas treats a NaN source value as null instead of propagating NaN through the recurrence.",
        variant="nan_as_null",
    ),
    PinningLimitation(
        "ema",
        ("pandas",),
        ("14-partition_keys-v__ema_3",),
        "NaN and None partition keys merge, sharing the EMA state across reference partitions.",
    ),
    PinningLimitation(
        "resample",
        ("pyarrow",),
        ("26-partition_keys-v__resample_1_hour_count", "27-partition_keys-v__resample_1_hour_count"),
        "Arrow groups -0.0 and +0.0 separately, creating two buckets instead of one.",
    ),
    PinningLimitation(
        "resample",
        ("pandas", "duckdb"),
        (
            "1-bucket-v__resample_1_hour_sum",
            "7-rows-v__resample_1_hour_sum",
            "24-partition_keys-v__resample_1_hour_sum",
        ),
        "Buckets or partitions are sorted instead of retaining their first-occurrence order.",
    ),
    PinningLimitation(
        "resample",
        ("pandas",),
        ("25-partition_keys-v__resample_1_hour_sum",),
        "NaN and None partition keys merge, collapsing two output buckets into one.",
    ),
    PinningLimitation(
        "resample",
        ("duckdb",),
        ("25-partition_keys-v__resample_1_hour_sum",),
        "The NaN partition is emitted before the first-seen None partition.",
    ),
    PinningLimitation(
        "resample",
        ("pandas",),
        (
            "15-nan-v__resample_1_hour_sum",
            "16-nan-v__resample_1_hour_mean",
            "17-nan-v__resample_1_hour_count",
            "20-nan-v__resample_1_hour_min",
            "21-nan-v__resample_1_hour_max",
        ),
        "Pandas skips or does not count NaN and returns null for an all-NaN min/max bucket.",
        variant="nan_as_null",
    ),
    PinningLimitation(
        "resample",
        ("polars_lazy",),
        ("27-partition_keys-v__resample_1_hour_count",),
        "The first-seen negative zero partition key loses its sign.",
    ),
)
