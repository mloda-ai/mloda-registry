"""Base class for filtered aggregation feature groups.

Filtered aggregation computes an aggregate over a partitioned group,
considering only rows where a specified column matches a given value.
The result is broadcast back to every row in the partition, regardless
of whether that row matched the filter condition. The output always has
the same number of rows as the input.

Pattern: ``{col}__{agg}_filtered_groupby``

Example: ``sales__sum_filtered_groupby`` with
``filter_column="category", filter_value="electronics"`` computes the
sum of ``sales`` only for rows where ``category == "electronics"``,
partitioned by a group key, and broadcasts the result to every row.
"""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.provider import DefaultOptionKeys, FeatureGroup

# Types allowed for filter_value in match_feature_group_criteria.
_ALLOWED_FILTER_VALUE_TYPES = (str, int, float, bool)


class FilteredAggregationFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Base class for filtered (conditional) aggregation operations that preserve row count.

    Filtered aggregation computes an aggregate over a partitioned group,
    considering only rows where a filter column matches a given value.
    The result is broadcast back to every row in that partition. The output
    always has the same number of rows as the input.

    This is equivalent to SQL's filtered window function:
    ``AGG(col) FILTER (WHERE filter_col = val) OVER (PARTITION BY ...)``,
    or equivalently:
    ``AGG(CASE WHEN filter_col = val THEN col END) OVER (PARTITION BY ...)``.

    ## Supported Aggregation Types

    - ``sum``: Sum of filtered values
    - ``avg``: Average of filtered values
    - ``count``: Count of non-null filtered values
    - ``min``: Minimum of filtered values
    - ``max``: Maximum of filtered values

    ## Feature Creation Methods

    ### 1. String-Based Creation

    Features follow the naming pattern:
    ``{source_column}__{aggregation_type}_filtered_groupby``

    Examples::

        features = [
            Feature("sales__sum_filtered_groupby", options=Options(context={
                "partition_by": ["region"],
                "filter_column": "category",
                "filter_value": "electronics",
            })),
            Feature("temperature__avg_filtered_groupby", options=Options(context={
                "partition_by": ["city"],
                "filter_column": "is_outdoor",
                "filter_value": True,
            })),
        ]

    ### 2. Configuration-Based Creation

    Uses Options with proper context parameter separation::

        feature = Feature(
            name="my_result",
            options=Options(
                context={
                    "aggregation_type": "sum",
                    "in_features": "sales",
                    "partition_by": ["region"],
                    "filter_column": "category",
                    "filter_value": "electronics",
                }
            ),
        )

    ## Parameter Classification

    ### Context Parameters
    - ``aggregation_type``: The type of aggregation to perform
    - ``in_features``: The source feature to aggregate
    - ``partition_by``: List of columns to partition by
    - ``filter_column``: Column to apply the filter condition on
    - ``filter_value``: Value to match in the filter column (str, int, float, or bool)
    """

    PREFIX_PATTERN = r".*__([\w]+)_filtered_groupby$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    AGGREGATION_TYPE = "aggregation_type"
    PARTITION_BY = "partition_by"
    FILTER_COLUMN = "filter_column"
    FILTER_VALUE = "filter_value"

    AGGREGATION_TYPES = {
        "sum": "Sum of filtered values",
        "avg": "Average of filtered values",
        "mean": "Average of filtered values",
        "count": "Count of non-null filtered values",
        "min": "Minimum of filtered values",
        "max": "Maximum of filtered values",
    }

    PROPERTY_MAPPING = {
        AGGREGATION_TYPE: {
            **AGGREGATION_TYPES,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature for filtered aggregation",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
        PARTITION_BY: {
            "explanation": "List of columns to partition by",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
        FILTER_COLUMN: {
            "explanation": "Column to apply the filter condition on",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
        FILTER_VALUE: {
            "explanation": "Value to match in the filter column",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
    }

    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, source_feature: str) -> bool:
        """Validate that the parsed aggregation type is in AGGREGATION_TYPES."""
        return operation_config in cls.AGGREGATION_TYPES

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Any,
        options: Any,
        _data_access_collection: Any = None,
    ) -> bool:
        """Extend mixin matching with partition_by, filter_column, and filter_value validation.

        The mixin (mloda >= 0.5.5) handles:
        - Pattern and config matching via PROPERTY_MAPPING
        - List-valued options (partition_by) via tuple conversion
        - MIN/MAX_IN_FEATURES enforcement

        We add:
        - partition_by type validation (must be a list of strings)
        - filter_column validation (must be a non-empty string)
        - filter_value validation (must be str, int, float, or bool)
        """
        if not super().match_feature_group_criteria(feature_name, options, _data_access_collection):
            return False

        partition_by = options.get(cls.PARTITION_BY)
        if not isinstance(partition_by, (list, tuple)):
            return False
        if not all(isinstance(item, str) for item in partition_by):
            return False

        filter_column = options.get(cls.FILTER_COLUMN)
        if not isinstance(filter_column, str):
            return False

        filter_value = options.get(cls.FILTER_VALUE)
        if not isinstance(filter_value, _ALLOWED_FILTER_VALUE_TYPES):
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

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Shared loop: extract params from each feature, delegate to _compute_filtered.

        Supports both string-based features (e.g. "value_int__sum_filtered_groupby")
        and configuration-based features (via Options with aggregation_type,
        in_features, partition_by, filter_column, filter_value).
        """
        table = data

        for feature in features.features:
            feature_name = feature.get_name()

            source_features = cls._extract_source_features(feature)
            source_col = source_features[0]
            agg_type = cls._extract_aggregation_type(feature)
            partition_by = feature.options.get(cls.PARTITION_BY)
            filter_column = feature.options.get(cls.FILTER_COLUMN)
            filter_value = feature.options.get(cls.FILTER_VALUE)

            table = cls._compute_filtered(
                table, feature_name, source_col, partition_by, agg_type, filter_column, filter_value
            )

        return table

    @classmethod
    def _compute_filtered(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        filter_column: str,
        filter_value: Any,
    ) -> Any:
        """Subclasses must implement the actual filtered aggregation computation."""
        raise NotImplementedError
