"""Base class for window aggregation feature groups."""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import FeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


# Dynamic aggregation type prefixes that support parameterized variants
_DYNAMIC_PREFIXES = ("percentile_", "ratio_to_")


class WindowAggregationFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Base class for window aggregation operations that preserve row count."""

    PREFIX_PATTERN = r".*__([\w]+)_groupby$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    AGGREGATION_TYPE = "aggregation_type"
    PARTITION_BY = "partition_by"

    AGGREGATION_TYPES = {
        "sum": "Sum of values",
        "avg": "Average of values",
        "count": "Count of non-null values",
        "min": "Minimum value",
        "max": "Maximum value",
        "std": "Standard deviation",
        "var": "Variance",
        "median": "Median value",
        "mode": "Most frequent value",
        "nunique": "Count of unique values",
        "first": "First value in partition",
        "last": "Last value in partition",
    }

    PROPERTY_MAPPING = {
        AGGREGATION_TYPE: {
            **AGGREGATION_TYPES,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature for window aggregation",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
        PARTITION_BY: {
            "explanation": "List of columns to partition by",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.validation_function: lambda v: (
                isinstance(v, list) and all(isinstance(item, str) for item in v)
            ),
        },
    }

    @classmethod
    def _supports_aggregation_type(cls, aggregation_type: str) -> bool:
        """Check if the given aggregation type is supported, including dynamic types."""
        if aggregation_type in cls.AGGREGATION_TYPES:
            return True
        for prefix in _DYNAMIC_PREFIXES:
            if aggregation_type.startswith(prefix):
                return True
        return False

    @classmethod
    def get_aggregation_type(cls, feature_name: str) -> str:
        """Extract the aggregation type from the feature name."""
        prefix_part, _ = FeatureChainParser.parse_feature_name(feature_name, [cls.PREFIX_PATTERN])
        if prefix_part is None:
            raise ValueError(f"Could not extract aggregation type from feature name: {feature_name}")
        return prefix_part

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: str | FeatureName,
        options: Options,
        _data_access_collection: Any = None,
    ) -> bool:
        """Match feature name and validate partition_by and aggregation type."""
        _feature_name = feature_name.name if isinstance(feature_name, FeatureName) else feature_name

        prefix_patterns = cls._get_prefix_patterns()

        # Check if the feature name matches the pattern
        operation_config, source_feature = FeatureChainParser.parse_feature_name(_feature_name, prefix_patterns)

        if operation_config is None or source_feature is None:
            return False

        # Validate the aggregation type
        if not cls._supports_aggregation_type(operation_config):
            return False

        # Validate partition_by is present and is a list of strings
        partition_by = options.get(cls.PARTITION_BY)
        if partition_by is None:
            return False
        if not isinstance(partition_by, list):
            return False
        if not all(isinstance(item, str) for item in partition_by):
            return False

        return True

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Shared loop: extract params from each feature, delegate to _compute_window."""
        table = data

        for feature in features.features:
            feature_name = feature.get_name()
            agg_type = cls.get_aggregation_type(feature_name)
            source_col = feature_name.rsplit("__", 1)[0]
            partition_by = feature.options.get("partition_by")

            table = cls._compute_window(table, feature_name, source_col, partition_by, agg_type)

        return table

    @classmethod
    def _compute_window(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
    ) -> Any:
        """Subclasses must implement the actual window computation."""
        raise NotImplementedError

    # -- Shared aggregation helpers --

    @classmethod
    def _aggregate(cls, values: list[Any], agg_type: str) -> Any:
        """Compute a single aggregate over a list of values (may contain None)."""
        non_null = [v for v in values if v is not None]

        if not non_null:
            return None

        if agg_type == "sum":
            return sum(non_null)
        if agg_type == "avg":
            return sum(non_null) / len(non_null)
        if agg_type == "count":
            return len(non_null)
        if agg_type == "min":
            return min(non_null)
        if agg_type == "max":
            return max(non_null)
        if agg_type == "std":
            return cls._std(non_null)
        if agg_type == "var":
            return cls._var(non_null)
        if agg_type == "median":
            return cls._median(non_null)
        if agg_type == "mode":
            return cls._mode(non_null)
        if agg_type == "nunique":
            return len(set(non_null))
        if agg_type == "first":
            return non_null[0]
        if agg_type == "last":
            return non_null[-1]

        raise ValueError(f"Unsupported aggregation type: {agg_type}")

    @classmethod
    def _std(cls, values: list[Any]) -> Any:
        if len(values) < 2:
            return None
        return cls._var(values) ** 0.5

    @classmethod
    def _var(cls, values: list[Any]) -> Any:
        if len(values) < 2:
            return None
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / (len(values) - 1)

    @classmethod
    def _median(cls, values: list[Any]) -> Any:
        s = sorted(values)
        n = len(s)
        mid = n // 2
        if n % 2 == 0:
            return (s[mid - 1] + s[mid]) / 2.0
        return float(s[mid])

    @classmethod
    def _mode(cls, values: list[Any]) -> Any:
        counts = Counter(values)
        return counts.most_common(1)[0][0]
