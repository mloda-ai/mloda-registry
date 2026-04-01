"""Base class for offset feature groups."""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.provider import DefaultOptionKeys, FeatureGroup


class OffsetFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Base class for offset operations that preserve row count.

    Offset operations access values at a fixed offset from the current row
    within an ordered partition. The output always has the same number of
    rows as the input.

    ## Supported Offset Types

    - ``lag_N``: Value N rows before the current row
    - ``lead_N``: Value N rows after the current row
    - ``diff_N``: Difference between current value and value N rows before
    - ``pct_change_N``: Percentage change from value N rows before
    - ``first_value``: First value in the partition
    - ``last_value``: Last value in the partition

    ## Feature Creation Methods

    ### 1. String-Based Creation

    Features follow the naming pattern: ``{source_column}__{offset_type}_offset``

    Examples::

        features = [
            Feature("sales__lag_1_offset", options=Options(context={
                "partition_by": ["region"], "order_by": "timestamp",
            })),
            Feature("price__diff_1_offset", options=Options(context={
                "partition_by": ["category"], "order_by": "date",
            })),
        ]

    ### 2. Configuration-Based Creation

    Uses Options with proper context parameter separation::

        feature = Feature(
            name="my_lag",
            options=Options(
                context={
                    "offset_type": "lag_1",
                    "in_features": "sales",
                    "partition_by": ["region"],
                    "order_by": "timestamp",
                }
            ),
        )

    ## Null Behavior

    Offset positions that fall outside the partition boundary produce null.
    For example, ``lag_1`` on the first row of a partition returns null.
    ``first_value`` and ``last_value`` ignore null source values; if all
    values in a partition are null, the result is null.

    ## Parameter Classification

    ### Context Parameters
    - ``offset_type``: The type of offset to perform
    - ``in_features``: The source feature
    - ``partition_by``: List of columns to partition by
    - ``order_by``: Column to order by within each partition
    """

    PREFIX_PATTERN = r".*__([\w]+)_offset$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    OFFSET_TYPE = "offset_type"
    PARTITION_BY = "partition_by"
    ORDER_BY = "order_by"

    OFFSET_TYPES = {
        "first_value": "First value in partition",
        "last_value": "Last value in partition",
    }

    PROPERTY_MAPPING = {
        OFFSET_TYPE: {
            **OFFSET_TYPES,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.type_validator: lambda v: isinstance(v, str),
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature for offset operation",
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
    def _supports_offset_type(cls, offset_type: str) -> bool:
        """Check if the given offset type is supported."""
        if offset_type in cls.OFFSET_TYPES:
            return True
        for prefix in ("lag_", "lead_", "diff_", "pct_change_"):
            if offset_type.startswith(prefix):
                suffix = offset_type[len(prefix) :]
                if suffix.isdigit() and int(suffix) >= 1:
                    return True
        return False

    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, source_feature: str) -> bool:
        """Validate that the offset type is supported."""
        return cls._supports_offset_type(operation_config)

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
        if not isinstance(partition_by, (list, tuple)):
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
    def get_offset_type(cls, feature_name: str) -> str:
        """Extract the offset type from a feature name string."""
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        raise ValueError(f"Could not extract offset type from feature name: {feature_name}")

    @classmethod
    def _extract_offset_type(cls, feature: Any) -> str:
        """Extract offset type from feature (string-based or config-based)."""
        feature_name = feature.get_name()
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        offset_type = feature.options.get(cls.OFFSET_TYPE)
        if offset_type is None:
            raise ValueError(f"Could not extract offset type for {feature_name}")
        return str(offset_type)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Shared loop: extract params from each feature, delegate to _compute_offset."""
        table = data

        for feature in features.features:
            feature_name = feature.get_name()

            source_features = cls._extract_source_features(feature)
            source_col = source_features[0]
            offset_type = cls._extract_offset_type(feature)
            partition_by = feature.options.get(cls.PARTITION_BY)
            order_by = feature.options.get(cls.ORDER_BY)

            table = cls._compute_offset(table, feature_name, source_col, partition_by, order_by, offset_type)

        return table

    @classmethod
    def _compute_offset(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
        offset_type: str,
    ) -> Any:
        """Subclasses must implement the actual offset computation."""
        raise NotImplementedError
