"""Base class for window aggregation feature groups."""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.provider import DefaultOptionKeys

from mloda.community.feature_groups.data_operations.aggregation_base import (
    AGGREGATION_TYPES as _BASE_AGGREGATION_TYPES,
    AggregationFeatureGroupBase,
)
from mloda.community.feature_groups.data_operations.mask_utils import MASK_KEY, parse_mask_spec

# Aggregation types that require an order_by column to be deterministic.
_ORDER_DEPENDENT_AGG_TYPES = {"first", "last"}

AGGREGATION_TYPES = {
    **_BASE_AGGREGATION_TYPES,
    "first": "First non-null value in ordered partition (requires order_by)",
    "last": "Last non-null value in ordered partition (requires order_by)",
}


class WindowAggregationFeatureGroup(AggregationFeatureGroupBase):
    """Base class for window aggregation operations that preserve row count.

    Window aggregation computes an aggregate over a partitioned group and
    broadcasts the result back to every row in that group. The output always
    has the same number of rows as the input.

    ## Supported Aggregation Types

    Order-independent (require ``partition_by`` only):

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

    Order-dependent (require both ``partition_by`` and ``order_by``):

    - ``first``: First non-null value in ordered partition
    - ``last``: Last non-null value in ordered partition

    ## Feature Creation Methods

    ### 1. String-Based Creation

    Features follow the naming pattern: ``{source_column}__{aggregation_type}_window``

    Examples::

        features = [
            Feature("sales__sum_window", options=Options(context={"partition_by": ["region"]})),
            Feature("temperature__avg_window", options=Options(context={"partition_by": ["city"]})),
            Feature("price__first_window", options=Options(context={
                "partition_by": ["region"], "order_by": "timestamp",
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
                }
            ),
        )

    ## Parameter Classification

    ### Context Parameters
    - ``aggregation_type``: The type of aggregation to perform
    - ``in_features``: The source feature to aggregate
    - ``partition_by``: List of columns to partition by
    - ``order_by``: Column to order by (required for first/last)
    """

    PREFIX_PATTERN = r".*__([\w]+)_window$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    PARTITION_BY = "partition_by"
    ORDER_BY = "order_by"

    AGGREGATION_TYPES = AGGREGATION_TYPES

    PROPERTY_MAPPING = {
        AggregationFeatureGroupBase.AGGREGATION_TYPE: {
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
            DefaultOptionKeys.strict_validation: False,
        },
        ORDER_BY: {
            "explanation": "Column to order by (required for first/last)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.default: None,
        },
        MASK_KEY: {
            "explanation": "Conditional mask: (column, operator, value) tuple or list of tuples",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.default: None,
        },
    }

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Any,
        options: Any,
        _data_access_collection: Any = None,
    ) -> bool:
        """Extend mixin matching with partition_by and order_by validation.

        The mixin (mloda >= 0.6.0) handles:
        - Pattern and config matching via PROPERTY_MAPPING
        - List-valued options (partition_by) via tuple conversion (#228)
        - MIN/MAX_IN_FEATURES enforcement (#231)

        We add:
        - partition_by type validation (must be a list of strings)
        - order_by required for first/last (must be a string)
        """
        if not super().match_feature_group_criteria(feature_name, options, _data_access_collection):
            return False

        partition_by = options.get(cls.PARTITION_BY)
        if not isinstance(partition_by, (list, tuple)):
            return False
        if not all(isinstance(item, str) for item in partition_by):
            return False

        # Determine the aggregation type to check if order_by is required
        agg_type = cls._resolve_agg_type(feature_name, options)
        if agg_type in _ORDER_DEPENDENT_AGG_TYPES:
            order_by = options.get(cls.ORDER_BY)
            if not isinstance(order_by, str):
                return False

        return True

    @classmethod
    def _resolve_agg_type(cls, feature_name: Any, options: Any) -> str | None:
        """Extract agg_type from feature name or options for validation."""
        name = str(feature_name)
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        agg_type = options.get(cls.AGGREGATION_TYPE)
        if agg_type is not None:
            return str(agg_type)
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Shared loop: extract params from each feature, delegate to _compute_window.

        Supports both string-based features (e.g. "value_int__sum_window") and
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
            order_by = feature.options.get(cls.ORDER_BY)
            mask_spec = parse_mask_spec(feature.options.get(MASK_KEY))

            table = cls._compute_window(table, feature_name, source_col, partition_by, agg_type, order_by, mask_spec)

        return table

    @classmethod
    def _compute_window(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        order_by: str | None = None,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> Any:
        """Subclasses must implement the actual window computation."""
        raise NotImplementedError
