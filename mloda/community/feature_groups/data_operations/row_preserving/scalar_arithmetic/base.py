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
error. The name path validates present option values and enforces
required presence on absent ones, but ``{col}__{op}_constant`` still
matches without an explicit ``constant`` because ``CONSTANT``'s
``property_spec(...)`` declares ``deferred_binding=True``: its real
value is supplied via ``Options`` and checked at compute time, not
parsed from the name. The missing-constant check then fires at compute
time with a clear error.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mloda.provider import (
    DefaultOptionKeys,
    FeatureSet,
    OperationFeatureGroup,
    SubtypeDeclaration,
    property_spec,
)
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.base import (
    OP_TOKEN_EXPECTED,
    SCALAR_NUMBER_EXPECTED,
    assert_source_columns_present,
    is_number_element,
    is_op_token,
    is_scalar_number,
    scalar_number_value,
)
from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.base import ArithmeticFeatureGroupBase

ARITHMETIC_OPERATIONS: dict[str, str] = {
    "add": "Element-wise addition of a constant",
    "subtract": "Element-wise subtraction of a constant",
    "multiply": "Element-wise multiplication by a constant",
    "divide": "Element-wise division by a constant",
}


class ScalarArithmeticFeatureGroup(ArithmeticFeatureGroupBase, OperationFeatureGroup):
    SUBTYPES = SubtypeDeclaration(key=ArithmeticFeatureGroupBase.ARITHMETIC_OP)

    PREFIX_PATTERN = r".*__([\w]+)_constant$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    OPERATION_LABEL = "scalar arithmetic"
    CONSTANT = "constant"

    PROPERTY_MAPPING = {
        ArithmeticFeatureGroupBase.ARITHMETIC_OP: property_spec(
            "Arithmetic operation applied between the source column and the constant",
            strict=True,
            allowed_values=ARITHMETIC_OPERATIONS,
            match_guard=is_op_token,
            expected=OP_TOKEN_EXPECTED,
        ),
        DefaultOptionKeys.in_features: property_spec(
            "Single source feature column for the arithmetic operation",
            strict=False,
        ),
        CONSTANT: property_spec(
            "Numeric constant applied element-wise to the source column",
            strict=True,
            element_validator=is_number_element,
            match_guard=is_scalar_number,
            expected=SCALAR_NUMBER_EXPECTED,
            deferred_binding=True,
        ),
    }

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
    def _non_numeric_descriptor(cls, data: Any, source_col: str) -> object | None:
        """Each Implementation passes its framework's check to ``apply_arithmetic``."""
        raise TypeError(
            f"{cls.__name__} has no framework check; each Implementation passes its own to apply_arithmetic."
        )

    @classmethod
    def apply_arithmetic(
        cls,
        data: Any,
        features: FeatureSet,
        *,
        non_numeric: Callable[[Any, str], object | None],
        compute_arithmetic: Callable[[Any, str, str, str, int | float], Any],
    ) -> Any:
        """Shared loop: one new column ``source {op} constant`` per feature, null source gives null.

        Divide-by-zero and a missing constant are rejected before dispatching to the backend.
        """
        table = data

        for feature in features.features:
            feature_name = feature.name

            source_col = cls._extract_single_source_feature(feature)
            assert_source_columns_present(data, [source_col])
            op = cls._extract_arithmetic_op(feature)

            descriptor = non_numeric(data, source_col)
            if descriptor is not None:
                cls._raise_non_numeric_source(source_col, descriptor)

            constant = cls._extract_constant(feature)

            table = compute_arithmetic(table, feature_name, source_col, op, constant)

        return table
