"""Base class for window aggregation feature groups."""

from __future__ import annotations

from collections import Counter
from typing import Any

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
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
    }

    @classmethod
    def _supports_aggregation_type(cls, aggregation_type: str) -> bool:
        """Check if the given aggregation type is supported, including dynamic types."""
        if aggregation_type in cls.AGGREGATION_TYPES:
            return True
        if aggregation_type.startswith("percentile_"):
            suffix = aggregation_type[len("percentile_") :]
            if suffix.isdigit() and 0 <= int(suffix) <= 100:
                return True
            return False
        if aggregation_type.startswith("ratio_to_"):
            target = aggregation_type[len("ratio_to_") :]
            return target in cls.AGGREGATION_TYPES
        return False

    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, source_feature: str) -> bool:
        """Validate that the aggregation type is supported (including dynamic types)."""
        return cls._supports_aggregation_type(operation_config)

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Any,
        options: Any,
        _data_access_collection: Any = None,
    ) -> bool:
        """Extend mixin matching with partition_by validation.

        The mixin handles pattern and config matching. We add partition_by
        validation here because list-valued options are not supported by
        the mixin's PROPERTY_MAPPING.
        """
        if not super().match_feature_group_criteria(feature_name, options, _data_access_collection):
            return False

        partition_by = options.get(cls.PARTITION_BY)
        if partition_by is None:
            return False
        if not isinstance(partition_by, list):
            return False
        if not all(isinstance(item, str) for item in partition_by):
            return False

        return True

    @classmethod
    def get_aggregation_type(cls, feature_name: str) -> str:
        """Extract the aggregation type from a feature name string."""
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        raise ValueError(f"Could not extract aggregation type from feature name: {feature_name}")

    @classmethod
    def _extract_aggregation_type(cls, feature: Any) -> str:
        """Extract aggregation type from feature (string-based or config-based)."""
        feature_name = feature.get_name()
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        agg_type = feature.options.get(cls.AGGREGATION_TYPE)
        if agg_type is None:
            raise ValueError(f"Could not extract aggregation type for {feature_name}")
        return str(agg_type)

    @staticmethod
    def _get_column_names(data: Any) -> set[str]:
        """Extract column names from any supported data type."""
        if hasattr(data, "column_names"):
            return set(data.column_names)  # pa.Table
        if hasattr(data, "collect_schema"):
            return set(data.collect_schema().names())  # pl.LazyFrame
        if hasattr(data, "columns"):
            cols = data.columns
            if isinstance(cols, list):
                return set(cols)  # DuckdbRelation, SqliteRelation, pd.DataFrame
            return set(cols)
        return set()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Shared loop: extract params from each feature, delegate to _compute_window.

        Supports both string-based features (e.g. "value_int__sum_groupby") and
        configuration-based features (via Options with aggregation_type, in_features,
        partition_by).
        """
        table = data

        _MAX_IDENTIFIER_LENGTH = 1024

        for feature in features.features:
            feature_name = feature.get_name()

            if len(feature_name) > _MAX_IDENTIFIER_LENGTH:
                raise ValueError(f"Feature name exceeds maximum length of {_MAX_IDENTIFIER_LENGTH} characters")

            source_features = cls._extract_source_features(feature)
            source_col = source_features[0]
            agg_type = cls._extract_aggregation_type(feature)
            partition_by = feature.options.get(cls.PARTITION_BY)

            available = cls._get_column_names(table)
            if available:
                if source_col not in available:
                    raise ValueError(
                        f"Source column '{source_col}' not found in data. Available columns: {sorted(available)}"
                    )
                for col in partition_by:
                    if col not in available:
                        raise ValueError(
                            f"Partition column '{col}' not found in data. Available columns: {sorted(available)}"
                        )

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
        if not values:
            return None
        counts = Counter(values)
        return counts.most_common(1)[0][0]
