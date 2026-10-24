"""Shared skeleton for the aggregation feature-group families.

The aggregation, window-aggregation, scalar-aggregate, and frame-aggregate
families share the same aggregation-type extraction machinery: they parse the
same aggregation names out of the feature name or options, validate them against
a canonical aggregation-type table, and declare INT64 for counting ops. Issue
#246 had this logic duplicated across the families' ``base.py`` files.

``AggregationFeatureGroupBase`` holds the identical parts, mirroring the
``ArithmeticFeatureGroupBase`` consolidation from issue #214. The families
subclass it and override ``AGGREGATION_TYPES`` (so each family advertises its
own supported set / descriptions) and ``_COUNTING_AGG_TYPES`` (which agg types
produce an integer count), plus the family-specific bits (operand count,
matching, PROPERTY_MAPPING). Per-backend computation lives in the backend
modules.
"""

from __future__ import annotations

from typing import Any

from mloda.provider import FeatureChainParser, FeatureChainParserMixin, FeatureGroup
from mloda.user import DataType, Feature, Options

from mloda.community.feature_groups.data_operations.base import (
    FRAME_TYPE,
    KeyColumnInputsMixin,
    is_own_option,
    name_matches_prefix,
    op_token_value,
)

AGGREGATION_TYPES: dict[str, str] = {
    "sum": "Sum of values",
    "avg": "Average of values",
    "mean": "Average of values",
    "count": "Count of non-null values",
    "min": "Minimum value",
    "max": "Maximum value",
    "std": "Population standard deviation (ddof=0)",
    "var": "Population variance (ddof=0)",
    "std_pop": "Population standard deviation (ddof=0, same as std)",
    "std_samp": "Sample standard deviation (ddof=1)",
    "var_pop": "Population variance (ddof=0, same as var)",
    "var_samp": "Sample variance (ddof=1)",
    "median": "Median value",
    "mode": "Most frequent value",
    "nunique": "Count of unique values",
    "first": "First value in group",
    "last": "Last value in group",
}


# Released aggregation-family leaves subclass this alone, so it keeps FeatureGroup.
class AggregationFeatureGroupBase(KeyColumnInputsMixin, FeatureChainParserMixin, FeatureGroup):
    AGGREGATION_TYPE = "aggregation_type"

    #: Canonical aggregation-type table. Subclasses override to advertise their
    #: own supported set / descriptions.
    AGGREGATION_TYPES: dict[str, str] = AGGREGATION_TYPES

    #: Aggregation types that produce an integer count. Subclasses override which
    #: agg types declare INT64 via ``return_data_type_rule``.
    _COUNTING_AGG_TYPES: frozenset[str] = frozenset({"count", "nunique"})

    #: Own options that make a config-path feature belong to another family.
    CONFIG_DECLINE_KEYS: tuple[str, ...] = (FRAME_TYPE,)

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Any,
        options: Any,
        _data_access_collection: Any = None,
    ) -> bool:
        """Decline a config-path feature carrying an own option of another family."""
        if not name_matches_prefix(cls, feature_name) and any(
            is_own_option(options, k) for k in cls.CONFIG_DECLINE_KEYS
        ):
            return False
        return super().match_feature_group_criteria(feature_name, options, _data_access_collection)

    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, source_feature: str) -> bool:
        """Validate that the parsed aggregation type is in AGGREGATION_TYPES."""
        return operation_config in cls.AGGREGATION_TYPES

    @classmethod
    def get_aggregation_type(cls, feature_name: str) -> str:
        """Extract the aggregation type from a feature name string."""
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        raise ValueError(f"Could not extract aggregation type from feature name: {feature_name}")

    @classmethod
    def _extract_aggregation_type(cls, feature: Feature) -> str:
        """Extract aggregation type from feature (string-based or config-based)."""
        feature_name = feature.name
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        agg_type = feature.options.get(cls.AGGREGATION_TYPE)
        if agg_type is None:
            raise ValueError(f"Could not extract aggregation type for {feature_name}")
        return op_token_value(agg_type)

    @classmethod
    def _resolve_agg_type(cls, feature_name: str, options: Options) -> str | None:
        """Resolve the aggregation type from the feature name or options; None if unresolvable."""
        try:
            operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, cls._get_prefix_patterns())
        except ValueError:
            return None
        if operation_config is not None:
            return operation_config
        agg_type = options.get(cls.AGGREGATION_TYPE)
        return None if agg_type is None else op_token_value(agg_type)

    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> DataType | None:
        """Declare INT64 for counting ops; other aggregates stay open."""
        agg_type = cls._extract_aggregation_type(feature)
        if agg_type in cls._COUNTING_AGG_TYPES:
            return DataType.INT64
        return None
