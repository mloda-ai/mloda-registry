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
(the backend-specific fill); the source-column presence guard is shared
(``assert_source_columns_present``).
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
from mloda.user import Feature, FeatureName, Options

from mloda.community.feature_groups.data_operations.base import (
    COLUMN_REF_EXPECTED,
    always_required,
    assert_source_columns_present,
    column_ref_value,
    is_column_ref,
)


class FfillFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Base class for forward-fill-by-time operations that preserve row count.

    ffill is a single-op operation (no op/unit matrix). All backends support it
    natively; there are no rejections of supported inputs.
    """

    PREFIX_PATTERN = r".*__ffill$"
    RECOGNITION_ONLY_PATTERN = True

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    PARTITION_BY = "partition_by"
    ORDER_BY = "order_by"

    PROPERTY_MAPPING = {
        DefaultOptionKeys.in_features: property_spec(
            "Single source column to forward-fill",
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

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        _feature_name = str(feature_name)

        prefix_patterns = self._get_prefix_patterns()
        _operation_config, source_feature = FeatureChainParser.parse_feature_name(_feature_name, prefix_patterns)

        if source_feature:
            return {Feature(source_feature)}

        in_features_set = options.get_in_features()
        return set(in_features_set)

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
            raise ValueError("ffill requires an 'order_by' column in Options context.")
        return column_ref_value(order_by)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Compute one ffill column per feature in ``features``."""
        table = data

        for feature in features.features:
            feature_name = feature.name

            source_col = cls._extract_single_source_feature(feature)
            assert_source_columns_present(data, [source_col])
            partition_by = cls._extract_partition_by(feature)
            order_by = cls._extract_order_by(feature)

            table = cls._compute_ffill(table, feature_name, source_col, partition_by, order_by)

        return table

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
