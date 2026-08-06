"""Base class for scalar aggregate feature groups.

Computes a scalar aggregate over a single source column and broadcasts
the result to every row (global aggregate broadcast). Supports DuckDB,
SQLite, Pandas, Polars, and PyArrow backends.

Pattern: ``{col}__{agg}_scalar``

Example: ``value_int__sum_scalar`` computes the sum of the ``value_int``
column and fills every row with that scalar result.
"""

from __future__ import annotations

from typing import Any, ClassVar

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import DefaultOptionKeys

from mloda.community.feature_groups.data_operations.aggregation_base import AggregationFeatureGroupBase
from mloda.community.feature_groups.data_operations.mask_utils import MASK_KEY, parse_mask_spec
from mloda.community.feature_groups.data_operations.base import SingleSourceMixin, is_op_token

AGGREGATION_TYPES = {
    "sum": "Sum of values",
    "min": "Minimum value",
    "max": "Maximum value",
    "avg": "Average (mean) of values",
    "mean": "Average (mean) of values",
    "count": "Count of non-null values",
    "std": "Population standard deviation (ddof=0)",
    "var": "Population variance (ddof=0)",
    "std_pop": "Population standard deviation (ddof=0, same as std)",
    "std_samp": "Sample standard deviation (ddof=1)",
    "var_pop": "Population variance (ddof=0, same as var)",
    "var_samp": "Sample variance (ddof=1)",
    "median": "Median value",
}


class ScalarAggregateFeatureGroup(AggregationFeatureGroupBase, SingleSourceMixin):
    PREFIX_PATTERN = r".*__([\w]+)_scalar$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    SOURCE_LABEL = "Scalar aggregate"
    ENFORCE_MAX_IN_FEATURES = True

    AGGREGATION_TYPES = AGGREGATION_TYPES

    # Scalar aggregation declares INT64 only for count (not nunique, which it does not support).
    _COUNTING_AGG_TYPES = frozenset({"count"})

    _SUPPORTED_AGG_TYPES: ClassVar[frozenset[str]] = frozenset(AGGREGATION_TYPES)

    PROPERTY_MAPPING = {
        AggregationFeatureGroupBase.AGGREGATION_TYPE: {
            "explanation": "Aggregation applied over the whole column",
            DefaultOptionKeys.allowed_values: AGGREGATION_TYPES,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.match_guard: is_op_token,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Single source feature column to aggregate",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
        MASK_KEY: {
            "explanation": "Conditional mask: (column, operator, value) tuple or list of tuples",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.default: None,
        },
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return self._single_source_input_features(options, feature_name)

    @classmethod
    def _extract_source_features(cls, feature: Feature) -> list[str]:
        return cls._single_source_features(feature)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Compute a scalar aggregate per source column and broadcast to all rows.

        Each feature in the feature set produces one new column containing the
        aggregated scalar value repeated for every row. Only a single source
        column per feature is supported (enforced by MAX_IN_FEATURES = 1).
        """
        table = data

        for feature in features.features:
            feature_name = feature.name

            source_features = cls._extract_source_features(feature)
            source_col = source_features[0]
            agg_type = cls._extract_aggregation_type(feature)
            mask_spec = parse_mask_spec(feature.options.get(MASK_KEY))

            table = cls._compute_aggregation(table, feature_name, source_col, agg_type, mask_spec)

        return table

    @classmethod
    def _compute_aggregation(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        agg_type: str,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> Any:
        raise NotImplementedError
