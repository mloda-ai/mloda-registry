"""Base class for exponential-moving-average (EMA) feature groups.

Computes an exponentially weighted mean of a value column over time. Within
each partition, rows are sorted by an ``order_by`` (time) column ascending,
then an exponentially weighted mean is accumulated::

    ema[i] = alpha * x[i] + (1 - alpha) * ema[i-1]

with ``alpha = 2 / (span + 1)``, ``adjust=False`` and nulls SKIPPED in the
recurrence (a null input leaves the running ema unchanged and produces a NULL
output for that row). The first non-null seeds the recurrence. The operation is
ROW-PRESERVING: the result has the same rows in the same original order as the
input, with one new ``{col}__ema_{span}`` column appended.

Pattern: ``{src}__ema_{span}`` where ``span`` is a positive integer.

Examples::

    "value__ema_2"     # EMA of ``value`` with span 2, within each partition
    "value__ema_3"     # EMA of ``value`` with span 3

Options context:

- ``order_by``: REQUIRED column to sort by (ascending) within each partition.
- ``partition_by``: OPTIONAL list of columns; default ``[]`` treats the whole
  table as a single partition.
- ``in_features``: the single source column (when not derivable from the name).

The ``span`` is passed DIRECTLY to the underlying library (pandas
``ewm(span=...)`` / polars ``ewm_mean(span=...)``); backends must NOT
pre-convert to alpha -- each library performs the identical ``span -> alpha``
mapping internally.

Only pandas and polars-lazy compute EMA natively. PyArrow, DuckDB and SQLite
have no native exponentially weighted compute and a Python emulation is
forbidden by the CFW-backend rule, so they ship no backend for EMA (absence).
Compute subclasses implement ``_compute_ema`` (the backend EWM) and
``_assert_source_column_present`` (the guard).
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


class EmaFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Base class for exponential-moving-average operations that preserve row count."""

    PREFIX_PATTERN = r".*__ema_\d+$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    PARTITION_BY = "partition_by"
    ORDER_BY = "order_by"

    PROPERTY_MAPPING = {
        DefaultOptionKeys.in_features: {
            "explanation": "Single source column to compute the EMA of",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
        PARTITION_BY: {
            "explanation": "List of columns to partition by (default: whole table as one partition)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
        ORDER_BY: {
            "explanation": "Column to order by (ascending) within each partition",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        _feature_name = str(feature_name)

        prefix_patterns = self._get_prefix_patterns()
        _operation_config, source_feature = FeatureChainParser.parse_feature_name(_feature_name, prefix_patterns)

        if source_feature:
            return {Feature(source_feature)}

        in_features_set = options.get_in_features()
        return set(in_features_set)

    @classmethod
    def _extract_source_features(cls, feature: Feature) -> list[str]:
        """Extract and validate the single source feature.

        Returns a one-element list containing the source column name. Raises
        ``ValueError`` if more than one source feature is found, since EMA
        supports at most one source column.
        """
        feature_name = feature.name
        prefix_patterns = cls._get_prefix_patterns()

        _operation_config, source_feature = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)

        if source_feature:
            return [source_feature]

        in_features_set = feature.options.get_in_features()
        source_names: list[str] = [str(f.name) for f in in_features_set]

        if len(source_names) < cls.MIN_IN_FEATURES:
            raise ValueError(
                f"ema requires at least {cls.MIN_IN_FEATURES} source feature, "
                f"but got {len(source_names)} (in_features is empty)."
            )

        if len(source_names) > cls.MAX_IN_FEATURES:
            raise ValueError(
                f"ema supports at most {cls.MAX_IN_FEATURES} source feature, but got {len(source_names)}: {source_names}"
            )

        return source_names

    @classmethod
    def _extract_span(cls, feature: Feature) -> int:
        """Parse the positive-integer span from the ``{col}__ema_{span}`` name."""
        name = feature.name
        try:
            span = int(name.rsplit("__ema_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Could not extract a positive integer span from feature name {name!r}.") from exc
        if span <= 0:
            raise ValueError(f"ema span must be a positive integer (span > 0), got {span} in {name!r}.")
        return span

    @classmethod
    def _extract_partition_by(cls, feature: Feature) -> list[str]:
        """Return ``partition_by`` as a list (defaulting to ``[]`` when absent)."""
        partition_by = feature.options.get(cls.PARTITION_BY)
        if partition_by is None:
            return []
        return list(partition_by)

    @classmethod
    def _extract_order_by(cls, feature: Feature) -> str:
        """Return the required ``order_by`` column."""
        order_by = feature.options.get(cls.ORDER_BY)
        if order_by is None:
            raise ValueError("ema requires an 'order_by' column in Options context.")
        return str(order_by)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Compute one EMA column per feature in ``features``."""
        table = data

        for feature in features.features:
            feature_name = feature.name

            source_features = cls._extract_source_features(feature)
            source_col = source_features[0]
            span = cls._extract_span(feature)
            partition_by = cls._extract_partition_by(feature)
            order_by = cls._extract_order_by(feature)

            cls._assert_source_column_present(table, source_col)

            table = cls._compute_ema(table, feature_name, source_col, span, partition_by, order_by)

        return table

    @classmethod
    def _assert_source_column_present(cls, data: Any, source_col: str) -> None:
        """Reject a missing source column with a clear ``ValueError`` (backend-specific)."""
        raise NotImplementedError

    @classmethod
    def _compute_ema(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        span: int,
        partition_by: list[str],
        order_by: str,
    ) -> Any:
        raise NotImplementedError
