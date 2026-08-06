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
(SAME NAME, bucket-start value) and the aggregate column named exactly
``{src}__resample_{n}_{unit}_{agg}``. Output row order is not guaranteed.

PyArrow is the cross-framework reference. Subclasses implement ``_compute_resample``
(the backend floor + group + aggregate) and the two presence guards.
"""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import DefaultOptionKeys, FeatureGroup

from mloda.community.feature_groups.data_operations.base import (
    PartitionedSourceMixin,
    RejectionReasonMixin,
    always_required,
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

_RESAMPLE_MARKER = "resample"


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


class ResampleFeatureGroup(RejectionReasonMixin, FeatureGroup, PartitionedSourceMixin):
    """Base class for resample operations that CHANGE the row count.

    Subclasses must implement ``_compute_resample`` (the backend-specific
    floor + group-by + aggregate) and the two presence guards.
    """

    PREFIX_PATTERN = r".*__resample_[1-9]\d*_(?:minute|hour|day)_(?:mean|sum|count|min|max)$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    SOURCE_LABEL = "resample"
    ENFORCE_MIN_IN_FEATURES = True
    ENFORCE_MAX_IN_FEATURES = True
    VALIDATE_IN_FEATURE_COUNT = False

    PARTITION_BY = "partition_by"
    TIME_COLUMN = "time_column"
    RESAMPLE_OP = "resample_op"

    PROPERTY_MAPPING = {
        DefaultOptionKeys.in_features: {
            "explanation": "Single source column to aggregate per bucket",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
        PARTITION_BY: {
            "explanation": "List of columns to partition by (default: whole table as one partition)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
        TIME_COLUMN: {
            "explanation": "Column to floor into fixed-freq buckets",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.match_guard: is_column_ref,
            DefaultOptionKeys.required_when: always_required,
        },
        RESAMPLE_OP: {
            "explanation": "Resample token '{n}_{unit}_{agg}' (e.g. '1_hour_mean') when the op is not encoded in the feature name.",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.match_guard: _is_valid_resample_op,
        },
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return self._single_source_input_features(options, feature_name)

    # -- Name / token parsing ----------------------------------------------

    @classmethod
    def _source_from_name(cls, feature_name: str) -> str | None:
        """Return the source column from a ``{src}__resample_...`` name, else None.

        Overrides the mixin's pattern-based read on purpose: splitting on the LAST
        ``__resample_`` marker lets invalid-unit / invalid-agg / n=0 names (e.g.
        ``value__resample_1_century_mean``) still yield the source column and the raw
        token, so ``_parse_resample_op`` can raise the SPECIFIC error.
        """
        marker = f"__{_RESAMPLE_MARKER}_"
        idx = feature_name.rfind(marker)
        # Intentionally stricter than _token_from_name: idx == 0 means the name
        # begins with the marker and carries no source column, so return None
        # (whereas _token_from_name still yields the trailing token).
        if idx <= 0:
            return None
        return feature_name[:idx]

    @classmethod
    def _token_from_name(cls, feature_name: str) -> str | None:
        """Return the raw ``{n}_{unit}_{agg}`` token from the name, else None."""
        marker = f"__{_RESAMPLE_MARKER}_"
        idx = feature_name.rfind(marker)
        if idx < 0:
            return None
        return feature_name[idx + len(marker) :]

    @classmethod
    def _extract_source_features(cls, feature: Feature) -> list[str]:
        return cls._single_source_features(feature)

    @classmethod
    def _extract_resample_op(cls, feature: Feature) -> str:
        """Extract the raw ``{n}_{unit}_{agg}`` token from the name or Options."""
        token = cls._token_from_name(feature.name)
        if token is not None:
            return token
        op = feature.options.get(cls.RESAMPLE_OP)
        if op is None:
            raise ValueError(f"Could not extract resample op for {feature.name}")
        return op_token_value(op)

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
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Compute one resample output per feature in ``features``."""
        table = data

        for feature in features.features:
            feature_name = feature.name

            source_features = cls._extract_source_features(feature)
            source_col = source_features[0]
            op_token = cls._extract_resample_op(feature)
            n, unit, agg = _parse_resample_op(op_token)
            partition_by = cls._extract_partition_by(feature)
            time_column = cls._extract_time_column(feature)

            cls._assert_time_column_present(table, time_column)
            cls._assert_source_column_present(table, source_col)

            table = cls._compute_resample(table, feature_name, source_col, time_column, partition_by, n, unit, agg)

        return table

    @classmethod
    def _assert_time_column_present(cls, data: Any, time_column: str) -> None:
        """Reject a missing time column with a clear ``ValueError`` (backend-specific)."""
        raise NotImplementedError

    @classmethod
    def _assert_source_column_present(cls, data: Any, source_col: str) -> None:
        """Reject a missing source column with a clear ``ValueError`` (backend-specific)."""
        raise NotImplementedError

    @classmethod
    def _compute_resample(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        time_column: str,
        partition_by: list[str],
        n: int,
        unit: str,
        agg: str,
    ) -> Any:
        raise NotImplementedError
