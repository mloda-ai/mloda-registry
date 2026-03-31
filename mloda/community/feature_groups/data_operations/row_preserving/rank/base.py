"""Base class for rank feature groups."""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.provider import DefaultOptionKeys, FeatureGroup


class RankFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Base class for rank operations that preserve row count.

    Rank operations assign a rank or position to each row within a
    partition, ordered by a specified column. The output always has
    the same number of rows as the input.

    ## Supported Rank Types

    - ``row_number``: Sequential position (1, 2, 3, ...), no ties
    - ``rank``: Standard rank with gaps for ties (1, 2, 2, 4, ...)
    - ``dense_rank``: Rank without gaps (1, 2, 2, 3, ...)
    - ``percent_rank``: Relative rank as fraction from 0.0 to 1.0
    - ``ntile_N``: Divide rows into N roughly equal buckets (1..N)

    ## Feature Creation Methods

    ### 1. String-Based Creation

    Features follow the naming pattern: ``{source_column}__{rank_type}_ranked``

    Examples::

        features = [
            Feature("sales__row_number_ranked", options=Options(context={
                "partition_by": ["region"], "order_by": "sales",
            })),
            Feature("score__dense_rank_ranked", options=Options(context={
                "partition_by": ["team"], "order_by": "score",
            })),
            Feature("value__ntile_4_ranked", options=Options(context={
                "partition_by": ["region"], "order_by": "value",
            })),
        ]

    ### 2. Configuration-Based Creation

    Uses Options with proper context parameter separation::

        feature = Feature(
            name="my_rank",
            options=Options(
                context={
                    "rank_type": "row_number",
                    "in_features": "sales",
                    "partition_by": ["region"],
                    "order_by": "sales",
                }
            ),
        )

    ## Parameter Classification

    ### Context Parameters
    - ``rank_type``: The type of ranking to perform
    - ``in_features``: The source feature (used for ordering)
    - ``partition_by``: List of columns to partition by
    - ``order_by``: Column to order by within each partition
    """

    PREFIX_PATTERN = r".*__([\w]+)_ranked$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    RANK_TYPE = "rank_type"
    PARTITION_BY = "partition_by"
    ORDER_BY = "order_by"

    RANK_TYPES = {
        "row_number": "Sequential position, no ties",
        "rank": "Standard rank with gaps for ties",
        "dense_rank": "Rank without gaps for ties",
        "percent_rank": "Relative rank as fraction (0.0 to 1.0)",
    }

    PROPERTY_MAPPING = {
        RANK_TYPE: {
            **RANK_TYPES,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature for rank ordering",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
        PARTITION_BY: {
            "explanation": "List of columns to partition by",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
        ORDER_BY: {
            "explanation": "Column to order by within each partition",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
    }

    @classmethod
    def _supports_rank_type(cls, rank_type: str) -> bool:
        """Check if the given rank type is supported, including ntile_N."""
        if rank_type in cls.RANK_TYPES:
            return True
        if rank_type.startswith("ntile_"):
            suffix = rank_type[len("ntile_") :]
            if suffix.isdigit() and int(suffix) >= 1:
                return True
        return False

    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, source_feature: str) -> bool:
        """Validate that the rank type is supported (including ntile_N)."""
        return cls._supports_rank_type(operation_config)

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Any,
        options: Any,
        _data_access_collection: Any = None,
    ) -> bool:
        """Extend mixin matching with partition_by, order_by, and in_features validation."""
        if not super().match_feature_group_criteria(feature_name, options, _data_access_collection):
            return False

        partition_by = options.get(cls.PARTITION_BY)
        if partition_by is None:
            return False
        if not isinstance(partition_by, (list, tuple)):
            return False
        if not partition_by:
            return False
        if not all(isinstance(item, str) for item in partition_by):
            return False

        order_by = options.get(cls.ORDER_BY)
        if order_by is None:
            return False
        if not isinstance(order_by, str):
            return False

        in_features_raw = options.get(DefaultOptionKeys.in_features)
        if in_features_raw is not None:
            in_features = options.get_in_features()
            if len(in_features) > cls.MAX_IN_FEATURES:
                return False

        return True

    @classmethod
    def get_rank_type(cls, feature_name: str) -> str:
        """Extract the rank type from a feature name string."""
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        raise ValueError(f"Could not extract rank type from feature name: {feature_name}")

    @classmethod
    def _extract_rank_type(cls, feature: Any) -> str:
        """Extract rank type from feature (string-based or config-based)."""
        feature_name = feature.get_name()
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        rank_type = feature.options.get(cls.RANK_TYPE)
        if rank_type is None:
            raise ValueError(f"Could not extract rank type for {feature_name}")
        return str(rank_type)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Shared loop: extract params from each feature, delegate to _compute_rank.

        Supports both string-based features (e.g. "value_int__row_number_ranked") and
        configuration-based features (via Options with rank_type, in_features,
        partition_by, order_by).
        """
        table = data

        for feature in features.features:
            feature_name = feature.get_name()

            rank_type = cls._extract_rank_type(feature)
            partition_by = feature.options.get(cls.PARTITION_BY)
            order_by = feature.options.get(cls.ORDER_BY)

            table = cls._compute_rank(table, feature_name, partition_by, order_by, rank_type)

        return table

    @classmethod
    def _compute_rank(
        cls,
        data: Any,
        feature_name: str,
        partition_by: list[str],
        order_by: str,
        rank_type: str,
    ) -> Any:
        """Subclasses must implement the actual rank computation."""
        raise NotImplementedError
