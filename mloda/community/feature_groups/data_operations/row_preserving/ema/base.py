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
Compute subclasses implement ``_compute_ema`` (the backend EWM) and
``_assert_source_column_present`` (the guard).
"""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import DefaultOptionKeys, FeatureGroup

from mloda.community.feature_groups.data_operations.base import (
    OrderedSourceMixin,
    RejectionReasonMixin,
    always_required,
    is_column_ref,
)


class EmaFeatureGroup(RejectionReasonMixin, FeatureGroup, OrderedSourceMixin):
    """Base class for exponential-moving-average operations that preserve row count."""

    PREFIX_PATTERN = r".*__ema_\d+$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    SOURCE_LABEL = "ema"
    ENFORCE_MIN_IN_FEATURES = True
    ENFORCE_MAX_IN_FEATURES = True
    VALIDATE_IN_FEATURE_COUNT = False

    PARTITION_BY = "partition_by"
    ORDER_BY = "order_by"

    PROPERTY_MAPPING = {
        DefaultOptionKeys.in_features: {
            "explanation": "Single source column to compute the EMA of",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
        PARTITION_BY: {
            "explanation": "List of columns to partition by (default: whole table as one partition)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
        ORDER_BY: {
            "explanation": "Column to order by (ascending) within each partition",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.match_guard: is_column_ref,
            DefaultOptionKeys.required_when: always_required,
        },
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return self._single_source_input_features(options, feature_name)

    @classmethod
    def _extract_source_features(cls, feature: Feature) -> list[str]:
        return cls._single_source_features(feature)

    @classmethod
    def _extract_span(cls, feature: Feature) -> int:
        """Parse the positive-integer span from the ``{col}__ema_{span}`` name."""
        name = feature.name
        try:
            span = int(name.rsplit("__ema_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Could not extract a positive integer span from feature name {name!r}.") from exc
        if span <= 0:
            raise ValueError(f"ema span must be a positive integer (span > 0), got {span} in {name!r}.")
        return span

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Compute one EMA column per feature in ``features``."""
        table = data

        for feature in features.features:
            feature_name = feature.name

            source_features = cls._extract_source_features(feature)
            source_col = source_features[0]
            span = cls._extract_span(feature)
            partition_by = cls._extract_partition_by(feature)
            order_by = cls._extract_order_by(feature)

            cls._assert_source_column_present(table, source_col)

            table = cls._compute_ema(table, feature_name, source_col, span, partition_by, order_by)

        return table

    @classmethod
    def _assert_source_column_present(cls, data: Any, source_col: str) -> None:
        """Reject a missing source column with a clear ``ValueError`` (backend-specific)."""
        raise NotImplementedError

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
