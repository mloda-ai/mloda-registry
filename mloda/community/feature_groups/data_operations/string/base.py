"""Base class for string operation feature groups."""

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

STRING_OPS = {
    "upper": "Convert string to uppercase",
    "lower": "Convert string to lowercase",
    "trim": "Strip leading and trailing whitespace",
    "length": "Return the length of the string (integer)",
    "reverse": "Reverse the string",
}


# FeatureGroup is listed so the mixin stays ahead of it in the MRO.
class StringFeatureGroup(OperationFeatureGroup, FeatureChainParserMixin, FeatureGroup):
    """Base class for element-wise string operations that preserve row count.

    String operations transform a single string column element by element.
    The output always has the same number of rows as the input.

    Supported operations: upper, lower, trim, length, reverse.

    Feature Creation Methods
    ------------------------

    1. String-based (pattern):
        Features follow the naming pattern ``<source_column>__<operation>``.

        Examples::

            Feature("name__upper")
            Feature("title__trim")
            Feature("description__length")

    2. Configuration-based:
        Uses Options with context parameters::

            Feature(
                "uppercased_name",
                options=Options(context={
                    "string_op": "upper",
                    "in_features": "name",
                }),
            )

    Reference Contract
    ------------------

    Null input gives null output; the PyArrow implementation is the reference. Row count and order are
    preserved. ``length`` declares INT64. Each implementation passes ``compute_string(table, feature_name,
    source_col, op)`` to ``transform_string``; it returns the input with the new column appended. SQLite serves only trim and length (``SqliteStringOps.supported``).
    """

    PREFIX_PATTERN = r".+__(upper|lower|trim|length|reverse)$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    STRING_OP = "string_op"

    SUBTYPES = SubtypeDeclaration(key=STRING_OP)

    PROPERTY_MAPPING = {
        STRING_OP: property_spec(
            "String operation applied to the source column",
            strict=True,
            allowed_values=STRING_OPS,
            match_guard=is_op_token,
            expected=OP_TOKEN_EXPECTED,
        ),
        DefaultOptionKeys.in_features: property_spec(
            "Source string column",
            strict=False,
        ),
    }

    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, source_feature: str) -> bool:
        """Validate that the parsed operation is a known string operation.

        Per-framework restriction is declared on the Implementation through
        ``Implementation.supported`` (e.g. SqliteStringOps), not here.
        """
        return operation_config in STRING_OPS

    @classmethod
    def get_string_op(cls, feature_name: str) -> str:
        """Extract the string operation from a pattern-based feature name."""
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        raise ValueError(f"Could not extract string operation from feature name: {feature_name}")

    @classmethod
    def _extract_string_op(cls, feature: Feature) -> str:
        """Extract string operation from feature (string-based or config-based)."""
        feature_name = feature.name
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        op = feature.options.get(cls.STRING_OP)
        if op is None:
            raise ValueError(f"Could not extract string operation for {feature_name}")
        return op_token_value(op)

    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> DataType | None:
        """Declare INT64 for length (an integer-valued op); other ops stay open."""
        op = cls._extract_string_op(feature)
        if op == "length":
            return DataType.INT64
        return None

    @classmethod
    def transform_string(
        cls,
        data: Any,
        features: FeatureSet,
        *,
        compute_string: Callable[[Any, str, str, str], Any],
    ) -> Any:
        """Shared loop: extract params from each feature, delegate to ``compute_string``."""
        table = data

        for feature in features.features:
            feature_name = feature.name

            source_col = cls._extract_single_source_feature(feature)
            assert_source_columns_present(data, [source_col])
            op = cls._extract_string_op(feature)

            table = compute_string(table, feature_name, source_col, op)

        return table
