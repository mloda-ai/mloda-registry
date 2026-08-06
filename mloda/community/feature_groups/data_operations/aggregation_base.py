"""Shared skeleton for the aggregation feature-group families.

The aggregation, window-aggregation, scalar-aggregate, and frame-aggregate families share the same
aggregation-type extraction machinery, so ``AggregationFeatureGroupBase`` holds it. The families
subclass it and override ``AGGREGATION_TYPES`` (each family advertises its own supported set) and
``_COUNTING_AGG_TYPES`` (which agg types produce an integer count), plus the family-specific bits
(operand count, matching, PROPERTY_MAPPING). Per-backend computation lives in the backend modules.
"""

from __future__ import annotations

from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import FeatureGroup

from mloda.community.feature_groups.data_operations.base import OpTypeAccessorMixin, RejectionReasonMixin
from mloda.community.feature_groups.data_operations.capability_hook import SubtypeCapabilityHook

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


class AggregationFeatureGroupBase(SubtypeCapabilityHook, RejectionReasonMixin, FeatureGroup, OpTypeAccessorMixin):
    AGGREGATION_TYPE = "aggregation_type"

    OP_TYPE_LABEL = "aggregation type"

    #: Canonical aggregation-type table. Subclasses override to advertise their
    #: own supported set / descriptions.
    AGGREGATION_TYPES: dict[str, str] = AGGREGATION_TYPES

    #: Aggregation types that produce an integer count. Subclasses override which
    #: agg types declare INT64 via ``return_data_type_rule``.
    _COUNTING_AGG_TYPES: frozenset[str] = frozenset({"count", "nunique"})

    @classmethod
    def _op_type_key(cls) -> str:
        """The option key the aggregation type falls back to, read live so an override of it applies."""
        return cls.AGGREGATION_TYPE

    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, source_feature: str) -> bool:
        """Validate that the parsed aggregation type is in AGGREGATION_TYPES."""
        return operation_config in cls.AGGREGATION_TYPES

    @classmethod
    def get_aggregation_type(cls, feature_name: str) -> str:
        """Extract the aggregation type from a feature name string."""
        return cls._extract_op_type(feature_name)

    @classmethod
    def _extract_aggregation_type(cls, feature: Feature) -> str:
        """Extract aggregation type from feature (string-based or config-based)."""
        return cls._extract_op_type(feature.name, feature.options)

    @classmethod
    def _resolve_agg_type(cls, feature_name: str, options: Options) -> str | None:
        """Resolve the aggregation type from the feature name or options; None if unresolvable."""
        return cls._resolve_op_type(feature_name, options)

    @classmethod
    def _capability_subtype(cls, feature_name: str, options: Options) -> str | None:
        return cls._resolve_agg_type(feature_name, options)

    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> DataType | None:
        """Declare INT64 for counting ops; other aggregates stay open."""
        agg_type = cls._extract_aggregation_type(feature)
        if agg_type in cls._COUNTING_AGG_TYPES:
            return DataType.INT64
        return None
