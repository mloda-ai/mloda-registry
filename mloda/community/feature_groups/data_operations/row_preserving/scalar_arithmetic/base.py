"""Base class for scalar arithmetic feature groups.

Computes an element-wise arithmetic operation (add, subtract, multiply,
divide) between a single source column and a numeric constant supplied
via ``Options(context={"constant": <value>})``. Supports DuckDB, SQLite,
Pandas, Polars, and PyArrow backends.

Pattern: ``{col}__{op}_constant``

Example: ``value_int__divide_constant`` with ``constant=2`` divides every
non-null value in ``value_int`` by 2.

The ``constant`` option carries ``strict_validation=False`` so that
pattern-only matches (``{col}__{op}_constant``) succeed without it; the
missing-constant check then fires at compute time with a clear error.
"""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import DefaultOptionKeys, FeatureGroup

ARITHMETIC_OPERATIONS: dict[str, str] = {
    "add": "Element-wise addition of a constant",
    "subtract": "Element-wise subtraction of a constant",
    "multiply": "Element-wise multiplication by a constant",
    "divide": "Element-wise division by a constant",
}


class ScalarArithmeticFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    PREFIX_PATTERN = r".*__([\w]+)_constant$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    ARITHMETIC_OP = "arithmetic_op"
    CONSTANT = "constant"

    PROPERTY_MAPPING = {
        ARITHMETIC_OP: {
            **ARITHMETIC_OPERATIONS,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Single source feature column for the arithmetic operation",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
        CONSTANT: {
            "explanation": "Numeric constant applied element-wise to the source column",
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

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        _feature_name = str(feature_name)

        prefix_patterns = self._get_prefix_patterns()
        operation_config, source_feature = FeatureChainParser.parse_feature_name(_feature_name, prefix_patterns)

        if operation_config is not None and source_feature is not None and source_feature:
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

        if operation_config is not None and source_feature is not None and source_feature:
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

            constant = feature.options.get(cls.CONSTANT)
            if constant is None:
                raise ValueError(f"Missing required option 'constant' for feature {feature_name!r}")
            if not isinstance(constant, (int, float)):
                raise ValueError(
                    f"Option 'constant' for feature {feature_name!r} must be int or float, "
                    f"got {type(constant).__name__}"
                )
            if op == "divide" and constant == 0:
                raise ValueError(f"Cannot divide by zero for feature {feature_name!r}")

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
