"""Base class for string operation feature groups."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mloda.provider import (
    DefaultOptionKeys,
    FeatureChainParser,
    FeatureChainParserMixin,
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

# Python str.strip() whitespace, which the reference trims.
TRIM_WHITESPACE: tuple[int, ...] = (
    0x09,
    0x0A,
    0x0B,
    0x0C,
    0x0D,
    0x1C,
    0x1D,
    0x1E,
    0x1F,
    0x20,
    0x85,
    0xA0,
    0x1680,
    0x2000,
    0x2001,
    0x2002,
    0x2003,
    0x2004,
    0x2005,
    0x2006,
    0x2007,
    0x2008,
    0x2009,
    0x200A,
    0x2028,
    0x2029,
    0x202F,
    0x205F,
    0x3000,
)


class StringFeatureGroup(FeatureChainParserMixin, OperationFeatureGroup):
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

    Null input gives null output; the pure Python reference in mloda-testing is the reference, and upper and lower
    use PyArrow's simple per-codepoint case mapping (utf8proc, ß -> ẞ). Row count and order are preserved. ``length``
    declares INT64. Each implementation passes ``compute_string(table, feature_name, source_col, op)`` to
    ``transform_string``; it returns the input with the new column appended. SQLite serves only trim and length
    (``SqliteStringOps.supported``).
    """

    CONTRACT_VERSION = 1

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
