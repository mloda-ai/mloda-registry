"""Base class for ffill-by-time feature groups.

Forward-fills a value column across time gaps. Within each partition, rows are
sorted by an ``order_by`` (time) column ascending, then the last non-null value
of the source column is carried FORWARD to fill nulls. The operation is
ROW-PRESERVING: the result has the same rows in the same original order as the
input, with one new ``{col}__ffill`` column appended.

Pattern: ``{src}__ffill``

Examples::

    "value__ffill"     # forward-fill ``value`` within each partition, by time

Options context:

- ``order_by``: REQUIRED column to sort by (ascending) within each partition.
- ``partition_by``: OPTIONAL list of columns; default ``[]`` treats the whole
  table as a single partition.
- ``in_features``: the single source column (when not derivable from the name).

Null rules pinned across all backends:

- Leading nulls (before the first non-null in time order) stay NULL.
- A null that follows a non-null gets the carried value.
- Non-null source values pass through unchanged.

PyArrow is the cross-framework reference. Subclasses implement ``_compute_ffill``
(the backend-specific fill) and ``_assert_source_column_present`` (the guard).
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


class FfillFeatureGroup(RejectionReasonMixin, FeatureGroup, OrderedSourceMixin):
    """Base class for forward-fill-by-time operations that preserve row count.

    ffill is a single-op operation (no op/unit matrix). All backends support it
    natively; there are no rejections of supported inputs.
    """

    PREFIX_PATTERN = r".*__ffill$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    SOURCE_LABEL = "ffill"
    ENFORCE_MIN_IN_FEATURES = True
    ENFORCE_MAX_IN_FEATURES = True
    VALIDATE_IN_FEATURE_COUNT = False

    PARTITION_BY = "partition_by"
    ORDER_BY = "order_by"

    PROPERTY_MAPPING = {
        DefaultOptionKeys.in_features: {
            "explanation": "Single source column to forward-fill",
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
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Compute one ffill column per feature in ``features``."""
        table = data

        for feature in features.features:
            feature_name = feature.name

            source_features = cls._extract_source_features(feature)
            source_col = source_features[0]
            partition_by = cls._extract_partition_by(feature)
            order_by = cls._extract_order_by(feature)

            cls._assert_source_column_present(table, source_col)

            table = cls._compute_ffill(table, feature_name, source_col, partition_by, order_by)

        return table

    @classmethod
    def _assert_source_column_present(cls, data: Any, source_col: str) -> None:
        """Reject a missing source column with a clear ``ValueError`` (backend-specific)."""
        raise NotImplementedError

    @classmethod
    def _compute_ffill(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
    ) -> Any:
        raise NotImplementedError
