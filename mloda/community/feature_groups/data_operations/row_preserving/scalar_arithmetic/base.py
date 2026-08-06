"""Base class for scalar arithmetic feature groups.

Computes an element-wise arithmetic operation (add, subtract, multiply,
divide) between a single source column and a numeric constant supplied
via ``Options(context={"constant": <value>})``. Supports DuckDB, SQLite,
Pandas, Polars, and PyArrow backends.

Pattern: ``{col}__{op}_constant``

Example: ``value_int__divide_constant`` with ``constant=2`` divides every
non-null value in ``value_int`` by 2.

The ``constant`` option is strict with an element validator, so a
wrong-typed value is reported as a rejection reason in the resolution
error. A pattern match skips property validation, so
``{col}__{op}_constant`` still matches without a constant; the
missing-constant check then fires at compute time with a clear error.
"""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import DefaultOptionKeys

from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.base import ArithmeticFeatureGroupBase
from mloda.community.feature_groups.data_operations.base import (
    SingleSourceMixin,
    is_number_element,
    is_op_token,
    is_scalar_number,
    scalar_number_value,
)

ARITHMETIC_OPERATIONS: dict[str, str] = {
    "add": "Element-wise addition of a constant",
    "subtract": "Element-wise subtraction of a constant",
    "multiply": "Element-wise multiplication by a constant",
    "divide": "Element-wise division by a constant",
}


class ScalarArithmeticFeatureGroup(ArithmeticFeatureGroupBase, SingleSourceMixin):
    PREFIX_PATTERN = r".*__([\w]+)_constant$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    SOURCE_LABEL = "Scalar arithmetic"
    ENFORCE_MAX_IN_FEATURES = True
    OPERATION_LABEL = "scalar arithmetic"
    CONSTANT = "constant"

    PROPERTY_MAPPING = {
        ArithmeticFeatureGroupBase.ARITHMETIC_OP: {
            "explanation": "Arithmetic operation applied between the source column and the constant",
            DefaultOptionKeys.allowed_values: ARITHMETIC_OPERATIONS,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.match_guard: is_op_token,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Single source feature column for the arithmetic operation",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
        CONSTANT: {
            "explanation": "Numeric constant applied element-wise to the source column",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.element_validator: is_number_element,
            DefaultOptionKeys.match_guard: is_scalar_number,
        },
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return self._single_source_input_features(options, feature_name)

    @classmethod
    def _extract_source_features(cls, feature: Feature) -> list[str]:
        return cls._single_source_features(feature)

    @classmethod
    def _extract_constant(cls, feature: Feature) -> int | float:
        """Return the constant as a bare number, unwrapped from its container.

        Owns the compute-time rejections: missing, non-numeric, and dividing by zero.
        """
        feature_name = feature.name

        constant = feature.options.get(cls.CONSTANT)
        if constant is None:
            raise ValueError(f"Missing required option 'constant' for feature {feature_name!r}")
        if not is_scalar_number(constant):
            raise ValueError(
                f"Option 'constant' for feature {feature_name!r} must be int or float, got {type(constant).__name__}"
            )

        value = scalar_number_value(constant)
        # Resolve the op only when it can matter, so a constant read never depends on the op path.
        if value == 0 and cls._extract_arithmetic_op(feature) == "divide":
            raise ValueError(f"Cannot divide by zero for feature {feature_name!r}")
        return value

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Compute an element-wise arithmetic operation per source column.

        Each feature produces one new column containing ``source {op} constant``.
        Null values in the source propagate to the result. Divide-by-zero and
        missing constant are rejected before dispatching to the backend.
        """
        table = data

        for feature in features.features:
            feature_name = feature.name

            source_features = cls._extract_source_features(feature)
            source_col = source_features[0]
            op = cls._extract_arithmetic_op(feature)

            cls._assert_source_column_is_numeric(data, source_col)

            constant = cls._extract_constant(feature)

            table = cls._compute_arithmetic(table, feature_name, source_col, op, constant)

        return table

    @classmethod
    def _compute_arithmetic(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        op: str,
        constant: int | float,
    ) -> Any:
        raise NotImplementedError
