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
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import DefaultOptionKeys, FeatureGroup

from mloda.community.feature_groups.data_operations.base import extract_partition_by

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


class ResampleFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Base class for resample operations that CHANGE the row count.

    Subclasses must implement ``_compute_resample`` (the backend-specific
    floor + group-by + aggregate) and the two presence guards.
    """

    PREFIX_PATTERN = r".*__resample_\d+_(?:minute|hour|day)_(?:mean|sum|count|min|max)$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

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
        },
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        source_feature = self._source_from_name(str(feature_name))
        if source_feature is not None:
            return {Feature(source_feature)}

        in_features_set = options.get_in_features()
        return set(in_features_set)

    # -- Name / token parsing ----------------------------------------------

    @classmethod
    def _source_from_name(cls, feature_name: str) -> str | None:
        """Return the source column from a ``{src}__resample_...`` name, else None.

        Permissive on purpose: it splits on the ``__resample_`` marker so that
        invalid-unit / invalid-agg / n=0 feature names (e.g.
        ``value__resample_1_century_mean``) still yield the source column and
        the raw token, letting ``_parse_resample_op`` raise the SPECIFIC error.
        """
        marker = f"__{_RESAMPLE_MARKER}_"
        idx = feature_name.find(marker)
        if idx <= 0:
            return None
        return feature_name[:idx]

    @classmethod
    def _token_from_name(cls, feature_name: str) -> str | None:
        """Return the raw ``{n}_{unit}_{agg}`` token from the name, else None."""
        marker = f"__{_RESAMPLE_MARKER}_"
        idx = feature_name.find(marker)
        if idx < 0:
            return None
        return feature_name[idx + len(marker) :]

    @classmethod
    def _extract_source_features(cls, feature: Feature) -> list[str]:
        """Extract and validate the single source feature.

        Name-based extraction is tried first; otherwise the source comes from
        ``in_features``. Raises ``ValueError`` if more than one source feature
        is supplied (MAX_IN_FEATURES=1).
        """
        source_feature = cls._source_from_name(feature.name)
        if source_feature is not None:
            return [source_feature]

        in_features_set = feature.options.get_in_features()
        source_names: list[str] = [str(f.name) for f in in_features_set]

        if len(source_names) < cls.MIN_IN_FEATURES:
            raise ValueError(
                f"resample requires at least {cls.MIN_IN_FEATURES} source feature, "
                f"but got {len(source_names)} (in_features is empty)."
            )

        if len(source_names) > cls.MAX_IN_FEATURES:
            raise ValueError(
                f"resample supports at most {cls.MAX_IN_FEATURES} source feature, but got {len(source_names)}: "
                f"{source_names}"
            )

        return source_names

    @classmethod
    def _extract_resample_op(cls, feature: Feature) -> str:
        """Extract the raw ``{n}_{unit}_{agg}`` token from the name or Options."""
        token = cls._token_from_name(feature.name)
        if token is not None:
            return token
        op = feature.options.get(cls.RESAMPLE_OP)
        if op is None:
            raise ValueError(f"Could not extract resample op for {feature.name}")
        return str(op)

    @classmethod
    def _extract_partition_by(cls, feature: Feature) -> list[str]:
        """Return ``partition_by`` as a list (defaulting to ``[]`` when absent)."""
        return extract_partition_by(feature, cls.PARTITION_BY)

    @classmethod
    def _extract_time_column(cls, feature: Feature) -> str:
        """Return the required ``time_column`` column."""
        time_column = feature.options.get(cls.TIME_COLUMN)
        if time_column is None:
            raise ValueError("resample requires a 'time_column' in Options context.")
        return str(time_column)

    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> DataType | None:
        """Declare INT64 for count buckets; other aggregations stay open."""
        try:
            op_token = cls._extract_resample_op(feature)
            _, _, agg = _parse_resample_op(op_token)
        except Exception:  # best-effort during planning; failure leaves the type undeclared
            return None
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
