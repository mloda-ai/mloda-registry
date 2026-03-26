"""Base class for datetime extraction feature groups."""

from __future__ import annotations

from typing import Any, List, Optional, Set

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import FeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

DATETIME_OPS = {
    "year": "Extract year from datetime",
    "month": "Extract month (1-12) from datetime",
    "day": "Extract day (1-31) from datetime",
    "hour": "Extract hour (0-23) from datetime",
    "minute": "Extract minute (0-59) from datetime",
    "second": "Extract second (0-59) from datetime",
    "dayofweek": "Day of week (0=Monday, 6=Sunday)",
    "is_weekend": "1 if Saturday/Sunday, 0 otherwise",
    "quarter": "Quarter of the year (1-4)",
}


class DateTimeFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    PREFIX_PATTERN = r".*__(year|month|day|hour|minute|second|dayofweek|is_weekend|quarter)$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    DATETIME_OP = "datetime_op"

    PROPERTY_MAPPING = {
        DATETIME_OP: {
            **DATETIME_OPS,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source datetime column",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
    }

    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, source_feature: str) -> bool:
        return operation_config in DATETIME_OPS

    @classmethod
    def get_datetime_op(cls, feature_name: str) -> str:
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        raise ValueError(f"Could not extract datetime operation from feature name: {feature_name}")

    @classmethod
    def _extract_datetime_op(cls, feature: Any) -> str:
        feature_name = feature.get_name()
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        op = feature.options.get(cls.DATETIME_OP)
        if op is None:
            raise ValueError(f"Could not extract datetime operation for {feature_name}")
        return str(op)

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        _feature_name = feature_name.name if isinstance(feature_name, FeatureName) else feature_name

        prefix_patterns = self._get_prefix_patterns()
        operation_config, source_feature = FeatureChainParser.parse_feature_name(_feature_name, prefix_patterns)

        if operation_config is not None and source_feature is not None and source_feature:
            return {Feature(source_feature)}

        in_features_set = options.get_in_features()
        self._validate_in_feature_count(list(in_features_set), _feature_name)
        return set(in_features_set)

    @classmethod
    def _extract_source_features(cls, feature: Feature) -> List[str]:
        feature_name = feature.get_name()
        prefix_patterns = cls._get_prefix_patterns()

        operation_config, source_feature = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)

        if operation_config is not None and source_feature is not None and source_feature:
            return [source_feature]

        in_features_set = feature.options.get_in_features()
        return [f.get_name() for f in in_features_set]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        table = data

        for feature in features.features:
            feature_name = feature.get_name()

            source_features = cls._extract_source_features(feature)
            source_col = source_features[0]
            op = cls._extract_datetime_op(feature)

            table = cls._compute_datetime(table, feature_name, source_col, op)

        return table

    @classmethod
    def _compute_datetime(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        op: str,
    ) -> Any:
        raise NotImplementedError
