"""Base class for datetime extraction feature groups."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mloda.provider import (
    DefaultOptionKeys,
    FeatureChainParser,
    FeatureChainParserMixin,
    FeatureGroup,
    FeatureSet,
    OperationFeatureGroup,
    SubtypeDeclaration,
    property_spec,
)
from mloda.user import DataType, Feature

from mloda.community.feature_groups.data_operations.base import (
    OP_TOKEN_EXPECTED,
    assert_source_columns_present,
    is_op_token,
    op_token_value,
)

DATETIME_OPS = {
    "year": "Extract year from datetime",
    "month": "Extract month (1-12) from datetime",
    "day": "Extract day (1-31) from datetime",
    "hour": "Extract hour (0-23) from datetime",
    "minute": "Extract minute (0-59) from datetime",
    "second": "Extract second (0-59) from datetime",
    "dayofweek": "Day of week (0=Monday, 6=Sunday)",
    "is_weekend": "1 if Saturday/Sunday, 0 otherwise",
    "quarter": "Quarter of the year (1-4)",
}


# FeatureGroup is listed so the mixin stays ahead of it in the MRO.
class DateTimeFeatureGroup(OperationFeatureGroup, FeatureChainParserMixin, FeatureGroup):
    """Base class for element-wise datetime extraction operations.

    Extracts scalar integer components from datetime columns. The output
    always has the same number of rows as the input (row-preserving).

    ## Supported Operations

    - ``year``: Four-digit year
    - ``month``: Month number (1-12)
    - ``day``: Day of month (1-31)
    - ``hour``: Hour (0-23)
    - ``minute``: Minute (0-59)
    - ``second``: Second (0-59)
    - ``dayofweek``: Day of week using Python convention (0=Monday, 6=Sunday)
    - ``is_weekend``: 1 if Saturday or Sunday, 0 otherwise
    - ``quarter``: Quarter of the year (1-4)

    ## Feature Creation Methods

    ### 1. Pattern-Based Creation

    Features follow the naming pattern: ``{source_column}__{operation}``

    Examples::

        features = [
            Feature("timestamp__year", options=Options()),
            Feature("created_at__dayofweek", options=Options()),
            Feature("event_time__is_weekend", options=Options()),
        ]

    ### 2. Configuration-Based Creation

    Uses Options with ``datetime_op`` and ``in_features`` context keys::

        feature = Feature(
            name="my_year_result",
            options=Options(
                context={
                    "datetime_op": "year",
                    "in_features": "timestamp",
                }
            ),
        )

    ## Reference Contract

    Null input gives null output; the PyArrow implementation is the reference. Row count and order are
    preserved. ``dayofweek`` uses 0=Monday, 6=Sunday. Every op declares INT64. Each implementation passes
    ``compute_datetime(data, feature_name, source_col, op)`` to the group; it returns the input with the new
    column appended.
    """

    PREFIX_PATTERN = r".*__(year|month|day|hour|minute|second|dayofweek|is_weekend|quarter)$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    DATETIME_OP = "datetime_op"

    SUBTYPES = SubtypeDeclaration(key=DATETIME_OP)

    PROPERTY_MAPPING = {
        DATETIME_OP: property_spec(
            "Datetime component extracted from the source column",
            strict=True,
            allowed_values=DATETIME_OPS,
            match_guard=is_op_token,
            expected=OP_TOKEN_EXPECTED,
        ),
        DefaultOptionKeys.in_features: property_spec(
            "Source datetime column",
        ),
    }

    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, source_feature: str) -> bool:
        return operation_config in DATETIME_OPS

    @classmethod
    def get_datetime_op(cls, feature_name: str) -> str:
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        raise ValueError(f"Could not extract datetime operation from feature name: {feature_name}")

    @classmethod
    def _extract_datetime_op(cls, feature: Feature) -> str:
        feature_name = feature.name
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        op = feature.options.get(cls.DATETIME_OP)
        if op is None:
            raise ValueError(f"Could not extract datetime operation for {feature_name}")
        return op_token_value(op)

    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> DataType | None:
        """Declare the deterministic output type: all datetime ops are integer-valued."""
        cls._extract_datetime_op(feature)
        return DataType.INT64

    @classmethod
    def extract_datetime(
        cls,
        data: Any,
        features: FeatureSet,
        *,
        compute_datetime: Callable[[Any, str, str, str], Any],
    ) -> Any:
        """Shared loop: extract params from each feature, delegate to ``compute_datetime``."""
        table = data

        for feature in features.features:
            feature_name = feature.name

            source_col = cls._extract_single_source_feature(feature)
            assert_source_columns_present(data, [source_col])
            op = cls._extract_datetime_op(feature)

            table = compute_datetime(table, feature_name, source_col, op)

        return table
