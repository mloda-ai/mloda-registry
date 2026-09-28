"""Base class for exponential-moving-average (EMA) feature groups.

Computes an exponentially weighted mean of a value column over time. Within
each partition, rows are sorted by an ``order_by`` (time) column ascending,
then an exponentially weighted mean is accumulated::

    ema[i] = alpha * x[i] + (1 - alpha) * ema[i-1]

with ``alpha = 2 / (span + 1)``, ``adjust=False`` and nulls SKIPPED in the
recurrence (a null input leaves the running ema unchanged and produces a NULL
output for that row). The first non-null seeds the recurrence. The operation is
ROW-PRESERVING: the result has the same rows in the same original order as the
input, with one new ``{col}__ema_{span}`` column appended.

Pattern: ``{src}__ema_{span}`` where ``span`` is a positive integer.

Examples::

    "value__ema_2"     # EMA of ``value`` with span 2, within each partition
    "value__ema_3"     # EMA of ``value`` with span 3

Options context:

- ``order_by``: REQUIRED column to sort by (ascending) within each partition.
- ``partition_by``: OPTIONAL list of columns; default ``[]`` treats the whole
  table as a single partition.
- ``in_features``: the single source column (when not derivable from the name).

The ``span`` is passed DIRECTLY to the underlying library (pandas
``ewm(span=...)`` / polars ``ewm_mean(span=...)``); backends must NOT
pre-convert to alpha -- each library performs the identical ``span -> alpha``
mapping internally.

Only pandas and polars-lazy compute EMA natively. PyArrow, DuckDB and SQLite
have no native exponentially weighted compute and a Python emulation is
forbidden by the CFW-backend rule, so they ship no backend for EMA (absence).
Compute subclasses implement ``_compute_ema`` (the backend EWM); the
source-column presence guard is shared (``assert_source_columns_present``).
"""

from __future__ import annotations

from typing import Any

from mloda.provider import (
    DefaultOptionKeys,
    FeatureChainParser,
    FeatureChainParserMixin,
    FeatureGroup,
    FeatureSet,
    property_spec,
)
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.base import (
    COLUMN_REF_EXPECTED,
    always_required,
    assert_key_columns_present,
    assert_source_columns_present,
    column_ref_value,
    is_column_ref,
)


class EmaFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Base class for exponential-moving-average operations that preserve row count."""

    PREFIX_PATTERN = r".*__ema_(\d+)$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    PARTITION_BY = "partition_by"
    ORDER_BY = "order_by"

    PROPERTY_MAPPING = {
        DefaultOptionKeys.in_features: property_spec(
            "Single source column to compute the EMA of",
        ),
        PARTITION_BY: property_spec(
            "List of columns to partition by (default: whole table as one partition)",
            default=None,
        ),
        ORDER_BY: property_spec(
            "Column to order by (ascending) within each partition",
            match_guard=is_column_ref,
            expected=COLUMN_REF_EXPECTED,
            required_when=always_required,
        ),
    }

    @classmethod
    def _extract_span(cls, feature: Feature) -> int:
        """Parse the positive-integer span from the ``{col}__ema_{span}`` name."""
        name = feature.name
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _source_feature = FeatureChainParser.parse_feature_name(name, prefix_patterns)

        if operation_config is None:
            raise ValueError(f"Could not extract a positive integer span from feature name {name!r}.")
        # PREFIX_PATTERN's capture group is \d+, so operation_config is all digits; int() cannot raise here.
        span = int(operation_config)
        if span <= 0:
            raise ValueError(f"ema span must be a positive integer (span > 0), got {span} in {name!r}.")
        return span

    @classmethod
    def _extract_partition_by(cls, feature: Feature) -> list[str]:
        """Return ``partition_by`` as a list (defaulting to ``[]`` when absent)."""
        partition_by = feature.options.get(cls.PARTITION_BY)
        if partition_by is None:
            return []
        return list(partition_by)

    @classmethod
    def _extract_order_by(cls, feature: Feature) -> str:
        """Return the required ``order_by`` column."""
        order_by = feature.options.get(cls.ORDER_BY)
        if order_by is None:
            raise ValueError("ema requires an 'order_by' column in Options context.")
        return column_ref_value(order_by)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Compute one EMA column per feature in ``features``."""
        table = data

        for feature in features.features:
            feature_name = feature.name

            source_col = cls._extract_single_source_feature(feature)
            assert_source_columns_present(data, [source_col])
            span = cls._extract_span(feature)
            partition_by = cls._extract_partition_by(feature)
            order_by = cls._extract_order_by(feature)
            assert_key_columns_present(data, partition_by, order_by)

            table = cls._compute_ema(table, feature_name, source_col, span, partition_by, order_by)

        return table

    @classmethod
    def _compute_ema(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        span: int,
        partition_by: list[str],
        order_by: str,
    ) -> Any:
        raise NotImplementedError
