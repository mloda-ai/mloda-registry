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

from mloda.provider import DefaultOptionKeys, FeatureSet, property_spec

from mloda.community.feature_groups.data_operations.base import (
    OP_TOKEN_EXPECTED,
    assert_source_columns_present,
    is_op_token,
)
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
    # The source side must carry the '&' separator: point arithmetic needs two
    # operands, so a one-operand name like 'x__add_point' cannot be computed.
    # Without the '&' here such a name matched at resolution time and only blew
    # up at compute time with a ValueError instead of a "no feature group found" error naming the real
    # problem. The config path (arithmetic_op plus a two-element in_features)
    # does not go through this pattern and is unaffected.
    PREFIX_PATTERN = r".*&.*__([\w]+)_point$"

    MIN_IN_FEATURES = 2
    MAX_IN_FEATURES = 2

    OPERATION_LABEL = "point arithmetic"

    PROPERTY_MAPPING = {
        ArithmeticFeatureGroupBase.ARITHMETIC_OP: property_spec(
            "Element-wise arithmetic operation applied to the two source columns",
            strict=True,
            allowed_values=ARITHMETIC_OPERATIONS,
            match_guard=is_op_token,
            expected=OP_TOKEN_EXPECTED,
        ),
        DefaultOptionKeys.in_features: property_spec(
            "Two source feature columns for the element-wise arithmetic operation",
            strict=False,
            match_guard=_is_ordered_in_features,
            expected="an ordered list or tuple of source features",
        ),
    }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Compute an element-wise arithmetic operation per pair of source columns.

        Each feature produces one new column containing ``col_a {op} col_b``.
        Null values in either source propagate to the result.
        """
        table = data

        for feature in features.features:
            feature_name = feature.name

            source_features = cls._extract_source_features(feature)
            cls.validate_in_feature_count(feature_name, len(source_features))
            col_a, col_b = source_features[0], source_features[1]

            assert_source_columns_present(data, [col_a, col_b])

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
