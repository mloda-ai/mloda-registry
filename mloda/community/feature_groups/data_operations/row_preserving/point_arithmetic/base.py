"""Base class for point arithmetic feature groups.

Computes an element-wise arithmetic operation (add, subtract, multiply,
divide) between two source columns. Supports DuckDB, SQLite, Pandas,
Polars, and PyArrow backends.

Pattern: ``{col_a}&{col_b}__{op}_point``

Example: ``value_int&amount__divide_point`` divides every row of
``value_int`` by the corresponding row of ``amount``.

Null values in either source column propagate to None in the output.
Divide-by-zero follows IEEE-754 float semantics on PyArrow/Pandas/Polars/
DuckDB (returns inf/nan) and returns NULL on SQLite.
"""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.provider import DefaultOptionKeys, FeatureGroup

from mloda.community.feature_groups.data_operations.reserved_columns import assert_no_reserved_columns

ARITHMETIC_OPERATIONS: dict[str, str] = {
    "add": "Element-wise addition of two columns",
    "subtract": "Element-wise subtraction of column b from column a",
    "multiply": "Element-wise multiplication of two columns",
    "divide": (
        "Element-wise division of column a by column b "
        "(null propagated from null operand; divide-by-zero follows IEEE-754 "
        "float semantics on PyArrow/Pandas/Polars/DuckDB, returns NULL on SQLite)"
    ),
}


class PointArithmeticFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    PREFIX_PATTERN = r".*__([\w]+)_point$"

    MIN_IN_FEATURES = 2
    MAX_IN_FEATURES = 2

    ARITHMETIC_OP = "arithmetic_op"

    PROPERTY_MAPPING = {
        ARITHMETIC_OP: {
            **ARITHMETIC_OPERATIONS,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Two source feature columns for the element-wise arithmetic operation",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
    }

    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, source_feature: str) -> bool:
        return operation_config in ARITHMETIC_OPERATIONS

    @classmethod
    def get_arithmetic_op(cls, feature_name: str) -> str:
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        raise ValueError(f"Could not extract arithmetic operation from feature name: {feature_name}")

    @classmethod
    def _extract_arithmetic_op(cls, feature: Feature) -> str:
        feature_name = feature.name
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        op = feature.options.get(cls.ARITHMETIC_OP)
        if op is None:
            raise ValueError(f"Could not extract arithmetic operation for {feature_name}")
        return str(op)

    @classmethod
    def _extract_source_features(cls, feature: Feature) -> list[str]:
        """Extract and validate the two source features for the arithmetic op.

        Returns a two-element list ``[col_a, col_b]`` preserving the order
        of the source columns as given in the feature name or options.
        Raises ValueError if the count is not exactly two, using the same
        wording as the mixin's ``_validate_in_feature_count``.
        """
        feature_name = feature.name
        prefix_patterns = cls._get_prefix_patterns()

        operation_config, source_feature = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)

        if operation_config and source_feature:
            source_names: list[str] = source_feature.split(cls.IN_FEATURE_SEPARATOR)
        else:
            # Read the raw in_features value to preserve order. Unordered
            # collections (set/frozenset) are rejected because operand order
            # is significant for non-commutative ops (subtract, divide).
            raw = feature.options.get(DefaultOptionKeys.in_features)
            if raw is None:
                source_names = []
            elif isinstance(raw, (list, tuple)):
                source_names = [str(item.name) if hasattr(item, "name") else str(item) for item in raw]
            elif isinstance(raw, str):
                source_names = [raw]
            elif isinstance(raw, (set, frozenset)):
                raise ValueError(
                    f"Feature '{feature_name}': in_features for point arithmetic must be an "
                    f"ordered list or tuple (got {type(raw).__name__}); operand order is "
                    f"significant for subtract and divide."
                )
            else:
                source_names = [str(item.name) if hasattr(item, "name") else str(item) for item in raw]

        count = len(source_names)
        if count < cls.MIN_IN_FEATURES:
            raise ValueError(
                f"Feature '{feature_name}' requires at least {cls.MIN_IN_FEATURES} in_feature(s), but found {count}"
            )
        if cls.MAX_IN_FEATURES is not None and count > cls.MAX_IN_FEATURES:
            raise ValueError(
                f"Feature '{feature_name}' allows at most {cls.MAX_IN_FEATURES} in_feature(s), but found {count}"
            )

        return source_names

    @staticmethod
    def _raise_non_numeric_source(source_col: str, got: object) -> None:
        """Shared error format for the numeric-source contract.

        Backend overrides of ``_assert_source_column_is_numeric`` call this
        helper so the message stays uniform across all backends. ``got`` is
        inlined verbatim, allowing backends to pass a native dtype, an
        affinity string, or any other descriptor.
        """
        raise ValueError(f"Source column {source_col!r} must be numeric for point arithmetic; got {got}.")

    @classmethod
    def _input_columns_and_framework(cls, data: Any) -> tuple[list[str], str]:
        """Return ``(column_names, framework_label)`` for ``data``.

        Backend-specific; implemented per backend so the base class has no
        compile-time or import-time dependency on any compute framework.
        """
        raise NotImplementedError

    @classmethod
    def _assert_source_column_is_numeric(cls, data: Any, source_col: str) -> None:
        """Reject non-numeric source columns with a clear ``ValueError``.

        Backend-specific; implemented per backend. Implementations should
        call ``cls._raise_non_numeric_source(source_col, <native dtype>)`` to
        keep the message format uniform.
        """
        raise NotImplementedError

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Compute an element-wise arithmetic operation per pair of source columns.

        Each feature produces one new column containing ``col_a {op} col_b``.
        Null values in either source propagate to the result.

        Reserved-column guard runs first so callers see the reserved-column
        error before any source-column validation.
        """
        column_names, framework_label = cls._input_columns_and_framework(data)
        assert_no_reserved_columns(column_names, framework=framework_label, operation="point arithmetic")

        table = data

        for feature in features.features:
            feature_name = feature.name

            source_features = cls._extract_source_features(feature)
            col_a, col_b = source_features[0], source_features[1]

            for source_col in (col_a, col_b):
                if source_col not in column_names:
                    raise ValueError(f"Source column {source_col!r} not found in input data")

            cls._assert_source_column_is_numeric(data, col_a)
            cls._assert_source_column_is_numeric(data, col_b)

            op = cls._extract_arithmetic_op(feature)

            table = cls._compute_arithmetic(table, feature_name, col_a, col_b, op)

        return table

    @classmethod
    def _compute_arithmetic(
        cls,
        data: Any,
        feature_name: str,
        col_a: str,
        col_b: str,
        op: str,
    ) -> Any:
        raise NotImplementedError
