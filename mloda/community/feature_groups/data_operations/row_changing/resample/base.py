"""Base class for resample feature groups.

Resample collapses event rows onto a regular time grid. Each row's
``time_column`` is floored to the start of its ``n*unit`` bucket (epoch-anchored
fixed-freq floor, IDENTICAL to ``time_bucketization``'s floor for
minute / hour / day so every backend agrees). Rows are then grouped by
``(*partition_by, bucket_start)`` and the source value column is aggregated with
one of the ORDER-INDEPENDENT aggregations ``mean / sum / count / min / max``.

This operation CHANGES the row count: the output has exactly one row per
NON-EMPTY ``(partition, bucket)`` pair. Empty gap buckets are NOT emitted. A
bucket that has rows but whose source values are ALL null still emits a row,
with ``count = 0`` and ``mean / sum / min / max = None`` (PyArrow oracle).

Pattern: ``{src}__resample_{n}_{unit}_{agg}``

Examples::

    "value__resample_1_hour_mean"     # hourly mean of ``value``
    "value__resample_15_minute_sum"   # 15-minute sum (epoch-anchored buckets)

Options context:

- ``time_column``: REQUIRED column to floor into buckets.
- ``partition_by``: OPTIONAL list of columns; default ``[]`` treats the whole
  table as a single partition.
- ``in_features``: the single source column (when not derivable from the name).

Output columns are the ``partition_by`` columns, the bucketed ``time_column``
(SAME NAME, bucket-start value) and one aggregate column per feature named exactly
``{src}__resample_{n}_{unit}_{agg}``. Output row order is not guaranteed.

PyArrow is the cross-framework reference. Features sharing one bucket grid are aggregated
together; a different grid needs its own request. Each backend is an ``Implementation``
supplying the floor + group + aggregate to ``resample``; presence guards are shared via
``assert_source_columns_present``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

from mloda.provider import (
    DefaultOptionKeys,
    FeatureChainParser,
    FeatureChainParserMixin,
    FeatureSet,
    OperationFeatureGroup,
    SubtypeDeclaration,
    property_spec,
)
from mloda.user import DataType, Feature, Options

from mloda.community.feature_groups.data_operations.base import (
    COLUMN_REF_EXPECTED,
    KeyColumnInputsMixin,
    always_required,
    assert_key_columns_present,
    assert_source_columns_present,
    column_ref_value,
    is_column_ref,
    is_op_token,
    op_token_value,
)

# Order-independent aggregations supported in v1. Order-dependent aggregations
# (e.g. ``first`` / ``last``) and ``median`` are deliberately excluded.
RESAMPLE_AGGS: dict[str, str] = {
    "mean": "Average of non-null values in the bucket",
    "sum": "Sum of non-null values (all-null bucket -> None)",
    "count": "Count of non-null values in the bucket",
    "min": "Minimum non-null value in the bucket",
    "max": "Maximum non-null value in the bucket",
}

# Fixed-freq units (epoch-anchored floor identical to time_bucketization).
RESAMPLE_UNITS: dict[str, str] = {
    "minute": "Minute-aligned buckets (sub-day, fixed length)",
    "hour": "Hour-aligned buckets (sub-day, fixed length)",
    "day": "Day-aligned buckets (calendar day, midnight UTC)",
}


class ResampleAgg(NamedTuple):
    """One aggregate to compute: output column ``feature_name`` = ``agg`` of ``source_col``."""

    feature_name: str
    source_col: str
    agg: str


def _parse_resample_op(token: str) -> tuple[int, str, str]:
    """Parse a resample token ``{n}_{unit}_{agg}`` into ``(n, unit, agg)``.

    Raises:
        ValueError: if the token is malformed, the unit or agg is unknown, or
            ``n`` is not a positive integer.
    """
    parts = token.split("_")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid resample token {token!r}: expected '{{n}}_{{unit}}_{{agg}}', "
            f"got {len(parts)} underscore-separated parts."
        )

    n_str, unit, agg = parts

    if unit not in RESAMPLE_UNITS:
        raise ValueError(f"Unsupported resample unit {unit!r} in {token!r}; supported: {sorted(RESAMPLE_UNITS)}.")

    if agg not in RESAMPLE_AGGS:
        raise ValueError(f"Unsupported resample agg {agg!r} in {token!r}; supported: {sorted(RESAMPLE_AGGS)}.")

    try:
        n = int(n_str)
    except ValueError as exc:
        raise ValueError(f"Resample bucket size in {token!r} must be a positive integer, got {n_str!r}.") from exc

    if n <= 0:
        raise ValueError(f"Resample bucket size n must be a positive integer (n > 0), got {n} in {token!r}.")

    return n, unit, agg


def _is_valid_resample_op(value: object) -> bool:
    """True when value is exactly one parseable '{n}_{unit}_{agg}' resample token, bare or in a container."""
    if not is_op_token(value):
        return False
    try:
        _parse_resample_op(op_token_value(value))
    except ValueError:
        return False
    return True


def _resolve_resample_agg_subtype(feature_name: str, options: Options) -> str | None:
    """Return the agg; core's safe_field turns a raise into None."""
    return _parse_resample_op(ResampleFeatureGroup._extract_resample_op(Feature(feature_name, options=options)))[2]


class ResampleFeatureGroup(KeyColumnInputsMixin, FeatureChainParserMixin, OperationFeatureGroup):
    """Base class for resample operations that CHANGE the row count.

    Each backend is an ``Implementation`` supplying the floor + group-by + aggregate to ``resample``;
    presence guards are shared.
    """

    SUBTYPES = SubtypeDeclaration(universe=RESAMPLE_AGGS.keys(), resolver=_resolve_resample_agg_subtype)

    KEY_COLUMN_OPTIONS = ("partition_by", "time_column")

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    PARTITION_BY = "partition_by"
    TIME_COLUMN = "time_column"
    RESAMPLE_OP = "resample_op"

    # Broad capture: 0 or a positive integer with no leading zero, then two lowercase-letter runs for
    # unit/agg. [a-z]+ (not \w+) keeps unit and agg disjoint from the "_" separator, so there is exactly
    # one way to split the token and no catastrophic backtracking. n=0 and an unknown unit/agg (e.g.
    # "century", "median") still match the pattern; _validate_string_match narrows via _parse_resample_op,
    # which already enumerates the valid units/aggs and rejects n<=0 (mirrors time_bucketization). Named
    # after RESAMPLE_OP so bind_name_captures can feed it into the forwarded-value mismatch check.
    PREFIX_PATTERN = rf".*__resample_(?P<{RESAMPLE_OP}>(?:[1-9]\d*|0)_[a-z]+_[a-z]+)$"

    PROPERTY_MAPPING = {
        DefaultOptionKeys.in_features: property_spec(
            "Single source column to aggregate per bucket",
        ),
        PARTITION_BY: property_spec(
            "List of columns to partition by (default: whole table as one partition)",
            default=None,
        ),
        TIME_COLUMN: property_spec(
            "Column to floor into fixed-freq buckets",
            match_guard=is_column_ref,
            expected=COLUMN_REF_EXPECTED,
            required_when=always_required,
        ),
        RESAMPLE_OP: property_spec(
            "Resample token '{n}_{unit}_{agg}' (e.g. '1_hour_mean') when the op is not encoded in the feature name.",
            match_guard=_is_valid_resample_op,
            expected="exactly one '{n}_{unit}_{agg}' resample token (e.g. '1_hour_mean')",
        ),
    }

    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, source_feature: str) -> bool:
        """Reject pattern matches whose token fails ``_parse_resample_op`` validation (bad unit/agg or n<=0)."""
        try:
            _parse_resample_op(operation_config)
        except ValueError:
            return False
        return True

    # -- Name / token parsing ----------------------------------------------

    @classmethod
    def _extract_resample_op(cls, feature: Feature) -> str:
        """Extract the raw ``{n}_{unit}_{agg}`` token from the name or Options."""
        prefix_patterns = cls._get_prefix_patterns()
        token, _source_feature = FeatureChainParser.parse_feature_name(feature.name, prefix_patterns)
        if token is not None:
            return token
        op = feature.options.get(cls.RESAMPLE_OP)
        if op is None:
            raise ValueError(f"Could not extract resample op for {feature.name}")
        return op_token_value(op)

    @classmethod
    def _extract_partition_by(cls, feature: Feature) -> list[str]:
        """Return ``partition_by`` as a list (defaulting to ``[]`` when absent)."""
        partition_by = feature.options.get(cls.PARTITION_BY)
        if partition_by is None:
            return []
        return list(partition_by)

    @classmethod
    def _extract_time_column(cls, feature: Feature) -> str:
        """Return the required ``time_column`` column."""
        time_column = feature.options.get(cls.TIME_COLUMN)
        if time_column is None:
            raise ValueError("resample requires a 'time_column' in Options context.")
        return column_ref_value(time_column)

    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> DataType | None:
        """Declare INT64 for count buckets; other aggregations stay open."""
        op_token = cls._extract_resample_op(feature)
        _, _, agg = _parse_resample_op(op_token)
        if agg == "count":
            return DataType.INT64
        return None

    @classmethod
    def resample(
        cls,
        data: Any,
        features: FeatureSet,
        *,
        compute_resample: Callable[[Any, list[ResampleAgg], str, list[str], int, str], Any],
    ) -> Any:
        """Compute all resample aggregates of ``features`` in one group-by.

        Raises:
            ValueError: if the features use more than one bucket grid (partition_by, time_column, n, unit).
        """
        aggs: list[ResampleAgg] = []
        grids: dict[tuple[tuple[str, ...], str, int, str], list[str]] = {}

        for feature in features.features:
            source_col = cls._extract_single_source_feature(feature)
            n, unit, agg = _parse_resample_op(cls._extract_resample_op(feature))
            partition_by = cls._extract_partition_by(feature)
            time_column = cls._extract_time_column(feature)

            assert_key_columns_present(data, partition_by, time_column, order_label="time_column")
            assert_source_columns_present(data, [source_col])

            aggs.append(ResampleAgg(feature.name, source_col, agg))
            grids.setdefault((tuple(partition_by), time_column, n, unit), []).append(feature.name)

        if len(grids) > 1:
            raise ValueError(
                f"Resample features with different bucket grids cannot share one request: {sorted(grids.values())}. "
                "Each bucket grid needs its own request."
            )

        ((partition_key, time_column, n, unit),) = grids
        return compute_resample(data, aggs, time_column, list(partition_key), n, unit)
