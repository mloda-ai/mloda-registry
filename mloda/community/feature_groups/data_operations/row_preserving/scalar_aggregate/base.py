"""Base class for scalar aggregate feature groups.

Computes a scalar aggregate over a single source column and broadcasts
the result to every row (global aggregate broadcast). Supports DuckDB,
SQLite, Pandas, Polars, and PyArrow backends.

Pattern: ``{col}__{agg}_scalar``

Example: ``value_int__sum_scalar`` computes the sum of the ``value_int``
column and fills every row with that scalar result.
"""

from __future__ import annotations

from typing import Any, List, Optional, Set

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import DefaultOptionKeys, FeatureGroup

AGGREGATION_TYPES = {
    "sum": "Sum of values",
    "min": "Minimum value",
    "max": "Maximum value",
    "avg": "Average (mean) of values",
    "mean": "Average (mean) of values",
    "count": "Count of non-null values",
    "std": "Standard deviation of values",
    "var": "Variance of values",
    "median": "Median value",
}


class ScalarAggregateFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    PREFIX_PATTERN = r".*__([\w]+)_scalar$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    AGGREGATION_TYPE = "aggregation_type"

    PROPERTY_MAPPING = {
        AGGREGATION_TYPE: {
            **AGGREGATION_TYPES,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Single source feature column to aggregate",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
    }

    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, source_feature: str) -> bool:
        return operation_config in AGGREGATION_TYPES

    @classmethod
    def get_aggregation_type(cls, feature_name: str) -> str:
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        raise ValueError(f"Could not extract aggregation type from feature name: {feature_name}")

    @classmethod
    def _extract_aggregation_type(cls, feature: Any) -> str:
        feature_name = feature.get_name()
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        agg_type = feature.options.get(cls.AGGREGATION_TYPE)
        if agg_type is None:
            raise ValueError(f"Could not extract aggregation type for {feature_name}")
        return str(agg_type)

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
        """Extract and validate the single source feature for aggregation.

        Returns a one-element list containing the source column name.
        Raises ValueError if more than one source feature is found, since
        this package only supports single-column aggregation.
        """
        feature_name = feature.get_name()
        prefix_patterns = cls._get_prefix_patterns()

        operation_config, source_feature = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)

        if operation_config is not None and source_feature is not None and source_feature:
            return [source_feature]

        in_features_set = feature.options.get_in_features()
        source_names = [f.get_name() for f in in_features_set]

        if len(source_names) > cls.MAX_IN_FEATURES:
            raise ValueError(
                f"Scalar aggregate supports at most {cls.MAX_IN_FEATURES} source feature, "
                f"but got {len(source_names)}: {source_names}"
            )

        return source_names

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Compute a scalar aggregate per source column and broadcast to all rows.

        Each feature in the feature set produces one new column containing the
        aggregated scalar value repeated for every row. Only a single source
        column per feature is supported (enforced by MAX_IN_FEATURES = 1).
        """
        table = data

        for feature in features.features:
            feature_name = feature.get_name()

            source_features = cls._extract_source_features(feature)
            source_col = source_features[0]
            agg_type = cls._extract_aggregation_type(feature)

            table = cls._compute_aggregation(table, feature_name, source_col, agg_type)

        return table

    @classmethod
    def _compute_aggregation(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        agg_type: str,
    ) -> Any:
        raise NotImplementedError
