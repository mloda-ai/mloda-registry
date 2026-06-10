"""Base class for point arithmetic feature groups.

Computes an element-wise arithmetic operation (add, subtract, multiply,
divide) between two source columns. Supports DuckDB, SQLite, Pandas,
Polars, and PyArrow backends.

Pattern: ``{col_a}&{col_b}__{op}_point``

Example: ``value_int&amount__divide_point`` divides every row of
``value_int`` by the corresponding row of ``amount``.

Null values in either source column propagate to None in the output.
``add``/``subtract``/``multiply`` use each backend's native operator
(int + int stays int, mixed int/float promotes); ``divide`` always
yields float (PyArrow/DuckDB/SQLite cast operands explicitly, while
Pandas/Polars rely on Python's ``/`` semantics). Divide-by-zero follows
IEEE-754 float semantics on PyArrow/Pandas/Polars/DuckDB (returns
inf/nan) and returns NULL on SQLite.
"""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.provider import DefaultOptionKeys

from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.base import ArithmeticFeatureGroupBase

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


def _is_ordered_in_features(value: object) -> bool:
    """Accept only ordered containers (list/tuple) for in_features.

    Operand order is significant for subtract and divide, so unordered
    collections (set/frozenset), mappings, and other iterables are
    rejected at match time rather than slipping through to fail (or
    behave unexpectedly) later in calculate_feature.
    """
    return isinstance(value, (list, tuple))


class PointArithmeticFeatureGroup(ArithmeticFeatureGroupBase):
    PREFIX_PATTERN = r".*__([\w]+)_point$"

    MIN_IN_FEATURES = 2
    MAX_IN_FEATURES = 2

    OPERATION_LABEL = "point arithmetic"

    PROPERTY_MAPPING = {
        ArithmeticFeatureGroupBase.ARITHMETIC_OP: {
            **ARITHMETIC_OPERATIONS,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Two source feature columns for the element-wise arithmetic operation",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.type_validator: _is_ordered_in_features,
        },
    }

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
            else:
                raise ValueError(
                    f"Feature '{feature_name}': in_features for point arithmetic must be an "
                    f"ordered list or tuple (got {type(raw).__name__}); operand order is "
                    f"significant for subtract and divide."
                )

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

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Compute an element-wise arithmetic operation per pair of source columns.

        Each feature produces one new column containing ``col_a {op} col_b``.
        Null values in either source propagate to the result.
        """
        column_names, _framework_label = cls._input_columns_and_framework(data)

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
