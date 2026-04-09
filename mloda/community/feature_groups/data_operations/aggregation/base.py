"""Base class for aggregation feature groups."""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.provider import DefaultOptionKeys, FeatureGroup

from mloda.community.feature_groups.data_operations.mask_utils import MASK_KEY, parse_mask_spec


class AggregationFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Base class for aggregation operations that reduce rows.

    Aggregation computes an aggregate over partitioned groups and
    returns one row per unique group key combination. The output has
    fewer rows than the input (one per group).

    Unlike window aggregation (``_window`` suffix), which broadcasts
    the aggregate back to every row (preserving the original row count),
    aggregation reduces the table to one row per group key combination.

    ## Supported Aggregation Types

    - ``sum``: Sum of values
    - ``avg``: Average of values
    - ``count``: Count of non-null values
    - ``min``: Minimum value
    - ``max``: Maximum value
    - ``std``: Population standard deviation (ddof=0)
    - ``var``: Population variance (ddof=0)
    - ``std_pop``: Population standard deviation (ddof=0, same as ``std``)
    - ``std_samp``: Sample standard deviation (ddof=1)
    - ``var_pop``: Population variance (ddof=0, same as ``var``)
    - ``var_samp``: Sample variance (ddof=1)
    - ``median``: Median value
    - ``mode``: Most frequent value
    - ``nunique``: Count of unique values
    - ``first``: First value in group
    - ``last``: Last value in group

    ## Feature Creation Methods

    ### 1. String-Based Creation

    Features follow the naming pattern: ``{source_column}__{aggregation_type}_agg``

    Examples::

        features = [
            Feature("sales__sum_agg", options=Options(context={"partition_by": ["region"]})),
            Feature("temperature__avg_agg", options=Options(context={"partition_by": ["city"]})),
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
                }
            ),
        )

    ## Parameter Classification

    ### Context Parameters
    - ``aggregation_type``: The type of aggregation to perform
    - ``in_features``: The source feature to aggregate
    - ``partition_by``: List of columns to partition by
    """

    PREFIX_PATTERN = r".*__([\w]+)_agg$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    AGGREGATION_TYPE = "aggregation_type"
    PARTITION_BY = "partition_by"

    AGGREGATION_TYPES = {
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

    PROPERTY_MAPPING = {
        AGGREGATION_TYPE: {
            **AGGREGATION_TYPES,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature for aggregation",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
        PARTITION_BY: {
            "explanation": "List of columns to partition by",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
        MASK_KEY: {
            "explanation": "Conditional mask: (column, operator, value) tuple or list of tuples",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.default: None,
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
        """Extend mixin matching with partition_by type validation.

        The mixin (mloda >= 0.5.5) handles:
        - Pattern and config matching via PROPERTY_MAPPING
        - List-valued options (partition_by) via tuple conversion (#228)
        - MIN/MAX_IN_FEATURES enforcement (#231)

        We add:
        - partition_by type validation (must be a list or tuple of strings)
        """
        if not super().match_feature_group_criteria(feature_name, options, _data_access_collection):
            return False

        partition_by = options.get(cls.PARTITION_BY)
        if not isinstance(partition_by, (list, tuple)):
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
        return str(agg_type)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Shared loop: extract params from each feature, delegate to _compute_group.

        Supports both string-based features (e.g. "value_int__sum_agg") and
        configuration-based features (via Options with aggregation_type, in_features,
        partition_by).
        """
        table = data

        for feature in features.features:
            feature_name = feature.name

            source_features = cls._extract_source_features(feature)
            source_col = source_features[0]
            agg_type = cls._extract_aggregation_type(feature)
            partition_by = feature.options.get(cls.PARTITION_BY)
            mask_spec = parse_mask_spec(feature.options.get(MASK_KEY))

            table = cls._compute_group(table, feature_name, source_col, partition_by, agg_type, mask_spec)

        return table

    @classmethod
    def _compute_group(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> Any:
        """Subclasses must implement the actual group computation."""
        raise NotImplementedError
