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
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import DefaultOptionKeys, property_spec

from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.base import ArithmeticFeatureGroupBase
from mloda.community.feature_groups.data_operations.base import (
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


class ScalarArithmeticFeatureGroup(ArithmeticFeatureGroupBase):
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
            deferred_binding=True,
        ),
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        _feature_name = str(feature_name)

        prefix_patterns = self._get_prefix_patterns()
        operation_config, source_feature = FeatureChainParser.parse_feature_name(_feature_name, prefix_patterns)

        if operation_config and source_feature:
            return {Feature(source_feature)}

        in_features_set = options.get_in_features()
        self._validate_in_feature_count(list(in_features_set), _feature_name)
        return set(in_features_set)

    @classmethod
    def _extract_source_features(cls, feature: Feature) -> list[str]:
        """Extract and validate the single source feature for the arithmetic op.

        Returns a one-element list containing the source column name.
        Raises ValueError if more than one source feature is found, since
        this package only supports single-column arithmetic.
        """
        feature_name = feature.name
        prefix_patterns = cls._get_prefix_patterns()

        operation_config, source_feature = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)

        if operation_config and source_feature:
            return [source_feature]

        in_features_set = feature.options.get_in_features()
        source_names: list[str] = [str(f.name) for f in in_features_set]

        if len(source_names) > cls.MAX_IN_FEATURES:
            raise ValueError(
                f"Scalar arithmetic supports at most {cls.MAX_IN_FEATURES} source feature, "
                f"but got {len(source_names)}: {source_names}"
            )

        return source_names

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
