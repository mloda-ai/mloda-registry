"""Base class for percentile feature groups."""

from __future__ import annotations

from typing import Any, List, Optional, Set

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import DefaultOptionKeys, FeatureGroup


class PercentileFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Base class for percentile operations that preserve row count.

    Computes a percentile over a partitioned group using PERCENTILE_CONT
    with linear interpolation and broadcasts the result back to every row
    in that group. The output always has the same number of rows as the input.

    ## Feature Creation Methods

    ### 1. String-Based Creation

    Features follow the naming pattern: ``{source_column}__p{N}_percentile``
    where N is an integer 0-100.

    Examples::

        features = [
            Feature("sales__p50_percentile", options=Options(context={"partition_by": ["region"]})),
            Feature("temperature__p95_percentile", options=Options(context={"partition_by": ["city"]})),
        ]

    ### 2. Configuration-Based Creation

        feature = Feature(
            name="my_result",
            options=Options(
                context={
                    "percentile": 0.75,
                    "in_features": "sales",
                    "partition_by": ["region"],
                }
            ),
        )
    """

    PREFIX_PATTERN = r".*__(p\d+)_percentile$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    PERCENTILE = "percentile"
    PARTITION_BY = "partition_by"

    PROPERTY_MAPPING = {
        PERCENTILE: {
            "explanation": "Percentile value (float between 0.0 and 1.0)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature for percentile computation",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
        PARTITION_BY: {
            "explanation": "List of columns to partition by",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
    }

    @classmethod
    def _parse_percentile_from_config(cls, operation_config: str) -> Optional[float]:
        """Parse a pN operation config into a float 0.0-1.0, or None if invalid."""
        n = int(operation_config[1:])
        if 0 <= n <= 100:
            return n / 100.0
        return None

    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, source_feature: str) -> bool:
        """Validate that the parsed percentile value is in the range 0-100."""
        return cls._parse_percentile_from_config(operation_config) is not None

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Any,
        options: Any,
        _data_access_collection: Any = None,
    ) -> bool:
        """Extend mixin matching with partition_by and percentile validation.

        The mixin handles:
        - Pattern and config matching via PROPERTY_MAPPING
        - List-valued options (partition_by) via tuple conversion
        - MIN/MAX_IN_FEATURES enforcement

        We add:
        - partition_by type validation (must be a list of strings)
        - percentile range validation for config-based features (0.0-1.0)
        """
        if not super().match_feature_group_criteria(feature_name, options, _data_access_collection):
            return False

        partition_by = options.get(cls.PARTITION_BY)
        if not isinstance(partition_by, (list, tuple)):
            return False
        if not all(isinstance(item, str) for item in partition_by):
            return False

        percentile = cls._resolve_percentile(feature_name, options)
        if percentile is None:
            return False
        if not isinstance(percentile, (int, float)):
            return False
        if percentile < 0.0 or percentile > 1.0:
            return False

        return True

    @classmethod
    def _resolve_percentile(cls, feature_name: Any, options: Any) -> Optional[float]:
        """Extract percentile as float from feature name or options."""
        name = str(feature_name)
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(name, prefix_patterns)
        if operation_config is not None:
            return cls._parse_percentile_from_config(operation_config)
        percentile = options.get(cls.PERCENTILE)
        if percentile is not None:
            if isinstance(percentile, (int, float)):
                return float(percentile)
        return None

    @classmethod
    def get_percentile_value(cls, feature_name: str) -> float:
        """Extract percentile float from a feature name string.

        Parses the ``pN`` portion from a feature name matching PREFIX_PATTERN
        and returns ``N / 100.0``.

        Raises ValueError if the feature name does not match.
        """
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            result = cls._parse_percentile_from_config(operation_config)
            if result is not None:
                return result
        raise ValueError(f"Could not extract percentile value from feature name: {feature_name}")

    @classmethod
    def _extract_percentile(cls, feature: Any) -> float:
        """Extract percentile float from feature (name first, then options)."""
        feature_name = feature.get_name()
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            result = cls._parse_percentile_from_config(operation_config)
            if result is not None:
                return result
        percentile = feature.options.get(cls.PERCENTILE)
        if percentile is None:
            raise ValueError(f"Could not extract percentile for {feature_name}")
        return float(percentile)

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Parse input features from feature name or options."""
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
        """Extract and validate the single source feature for percentile.

        Returns a one-element list containing the source column name.
        Raises ValueError if more than one source feature is found, since
        this package only supports single-column percentile computation.
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
                f"Percentile supports at most {cls.MAX_IN_FEATURES} source feature, "
                f"but got {len(source_names)}: {source_names}"
            )

        return source_names

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Compute a percentile per source column, partitioned, and broadcast to all rows.

        Each feature in the feature set produces one new column containing the
        percentile value repeated for every row in the partition.
        """
        table = data

        for feature in features.features:
            feature_name = feature.get_name()

            source_features = cls._extract_source_features(feature)
            source_col = source_features[0]
            percentile = cls._extract_percentile(feature)
            partition_by = feature.options.get(cls.PARTITION_BY)

            table = cls._compute_percentile(table, feature_name, source_col, partition_by, percentile)

        return table

    @classmethod
    def _compute_percentile(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        percentile: float,
    ) -> Any:
        """Subclasses must implement the actual percentile computation."""
        raise NotImplementedError
