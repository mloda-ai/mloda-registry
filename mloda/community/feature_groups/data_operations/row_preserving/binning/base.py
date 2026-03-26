"""Base class for binning operation feature groups."""

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

BINNING_OPS = {
    "bin": "Equal-width binning (pd.cut equivalent)",
    "qbin": "Quantile-based binning (pd.qcut equivalent)",
}


class BinningFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    PREFIX_PATTERN = r".*__(bin|qbin)_\d+$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    BINNING_OP = "binning_op"
    N_BINS = "n_bins"

    PROPERTY_MAPPING = {
        BINNING_OP: {
            **BINNING_OPS,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        N_BINS: {
            "explanation": "Number of bins (positive integer)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source numeric column",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
    }

    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, source_feature: str) -> bool:
        return operation_config in BINNING_OPS

    @classmethod
    def get_binning_params(cls, feature_name: str) -> tuple[str, int]:
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, source_feature = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None and source_feature is not None:
            n_bins = int(feature_name.rsplit("_", 1)[-1])
            return operation_config, n_bins
        raise ValueError(f"Could not extract binning parameters from feature name: {feature_name}")

    @classmethod
    def _extract_binning_params(cls, feature: Any) -> tuple[str, int]:
        feature_name = feature.get_name()
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, source_feature = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            n_bins = int(feature_name.rsplit("_", 1)[-1])
            return operation_config, n_bins
        op = feature.options.get(cls.BINNING_OP)
        n = feature.options.get(cls.N_BINS)
        if op is None or n is None:
            raise ValueError(f"Could not extract binning parameters for {feature_name}")
        return str(op), int(n)

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
            op, n_bins = cls._extract_binning_params(feature)

            table = cls._compute_binning(table, feature_name, source_col, op, n_bins)

        return table

    @classmethod
    def _compute_binning(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        op: str,
        n_bins: int,
    ) -> Any:
        raise NotImplementedError
