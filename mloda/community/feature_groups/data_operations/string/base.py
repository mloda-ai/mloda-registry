"""Base class for string operation feature groups."""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.provider import DefaultOptionKeys, FeatureGroup

STRING_OPS = {
    "upper": "Convert string to uppercase",
    "lower": "Convert string to lowercase",
    "trim": "Strip leading and trailing whitespace",
    "length": "Return the length of the string (integer)",
    "reverse": "Reverse the string",
}


class StringFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Base class for element-wise string operations that preserve row count.

    String operations transform a single string column element by element.
    The output always has the same number of rows as the input.

    Supported operations: upper, lower, trim, length, reverse.

    Not every compute framework supports every operation. For example,
    SQLite has no native ``REVERSE`` function, so ``SqliteStringOps``
    restricts its supported set via ``_validate_string_match``.
    SQLite's ``UPPER``/``LOWER`` only handle ASCII characters;
    non-ASCII accented characters are not transformed.

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
    """

    PREFIX_PATTERN = r".+__(upper|lower|trim|length|reverse)$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    STRING_OP = "string_op"

    PROPERTY_MAPPING = {
        STRING_OP: {
            **STRING_OPS,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source string column",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
    }

    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, source_feature: str) -> bool:
        """Validate that the parsed operation is a known string operation.

        Subclasses (e.g. SqliteStringOps) can override to further restrict
        the set of supported operations.
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
    def _extract_string_op(cls, feature: Any) -> str:
        """Extract string operation from feature (string-based or config-based)."""
        feature_name = feature.get_name()
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        op = feature.options.get(cls.STRING_OP)
        if op is None:
            raise ValueError(f"Could not extract string operation for {feature_name}")
        return str(op)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Shared loop: extract params from each feature, delegate to _compute_string."""
        table = data

        for feature in features.features:
            feature_name = feature.get_name()

            source_features = cls._extract_source_features(feature)
            source_col = source_features[0]
            op = cls._extract_string_op(feature)

            table = cls._compute_string(table, feature_name, source_col, op)

        return table

    @classmethod
    def _compute_string(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        op: str,
    ) -> Any:
        """Subclasses must implement the actual string computation."""
        raise NotImplementedError
