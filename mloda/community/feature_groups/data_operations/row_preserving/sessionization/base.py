"""Base class for sessionization feature groups.

``sessionize`` assigns an integer SESSION ID to each row by a gap-threshold
rule over a timestamp column. The source / in-feature column IS the timestamp
being sessionized. Within each partition, rows are sorted by an ``order_by``
(time) column ascending; a row STARTS A NEW SESSION when it is the first in
its partition OR the gap to the previous row (in time order, within the
partition) is STRICTLY GREATER than the threshold::

    is_new[i]    = first-in-partition OR (gap_to_previous > threshold)
    session_id   = cumsum(is_new) - 1   (over the sorted frame)

An equal gap (``gap == threshold``) stays in the SAME session: the rule is the
strict ``gap > threshold``. The session id is a GLOBALLY-UNIQUE 0-based integer
(ids are not reset per partition; they are unique across the whole sorted
frame). The operation is ROW-PRESERVING: the result has the same rows in the
same original order as the input, with one new
``{ts}__sessionize_{n}_{unit}`` int64 column appended.

Pattern: ``{ts}__sessionize_{n}_{unit}`` where ``n`` is a positive integer and
``unit`` is one of ``minute`` / ``hour`` / ``day`` / ``week``. The threshold is
``n`` of ``unit`` expressed in seconds (minute=60, hour=3600, day=86400,
week=604800).

Examples::

    "ts__sessionize_30_minute"     # new session when gap > 30 minutes
    "ts__sessionize_1_hour"        # new session when gap > 1 hour

Options context:

- ``order_by``: OPTIONAL column to sort by (ascending) within each partition;
  DEFAULTS to the named timestamp source column when absent.
- ``partition_by``: OPTIONAL list of columns; default ``[]`` treats the whole
  table as a single stream.
- ``in_features``: the single source column (when not derivable from the name).

Every backend (pandas, polars-lazy, PyArrow, DuckDB, SQLite) computes
sessionization NATIVELY; there is no rejection of supported inputs. PyArrow is
the cross-framework reference oracle. Compute subclasses implement
``_compute_session`` (the backend gap-and-cumsum) and
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

# Supported sessionization units mapped to their length in seconds. The four
# keys also define the units accepted by the feature-name regex.
SESSIONIZATION_UNITS: dict[str, int] = {
    "minute": 60,
    "hour": 3600,
    "day": 86400,
    "week": 604800,
}


def _parse_sessionize_op(token: str) -> tuple[int, str]:
    """Parse a ``sessionize_{n}_{unit}`` token into ``(n, unit)``.

    ``token`` is the operation portion of the feature name (everything from
    ``sessionize_`` onward, e.g. ``"sessionize_30_minute"``). Validates that
    ``n`` is a positive integer and ``unit`` is one of the supported units.
    """
    body = token.rsplit("sessionize_", 1)[-1]
    n_str, _, unit = body.partition("_")

    try:
        n = int(n_str)
    except ValueError as exc:
        raise ValueError(
            f"sessionize threshold n must be a positive integer (n > 0), got {n_str!r} in {token!r}."
        ) from exc
    if n <= 0:
        raise ValueError(f"sessionize threshold n must be a positive integer (n > 0), got {n} in {token!r}.")

    if unit not in SESSIONIZATION_UNITS:
        raise ValueError(
            f"Unsupported sessionization unit {unit!r}; expected one of {', '.join(SESSIONIZATION_UNITS)}."
        )

    return n, unit


def _sessionize_threshold_seconds(n: int, unit: str) -> int:
    """Return the gap threshold in seconds for ``n`` of ``unit``."""
    if n <= 0:
        raise ValueError(f"sessionize threshold n must be a positive integer (n > 0), got {n}.")
    if unit not in SESSIONIZATION_UNITS:
        raise ValueError(
            f"Unsupported sessionization unit {unit!r}; expected one of {', '.join(SESSIONIZATION_UNITS)}."
        )
    return n * SESSIONIZATION_UNITS[unit]


class SessionizationFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Base class for gap-threshold sessionization operations that preserve row count."""

    PREFIX_PATTERN = r".*__sessionize_\d+_(?:minute|hour|day|week)$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    PARTITION_BY = "partition_by"
    ORDER_BY = "order_by"

    PROPERTY_MAPPING = {
        DefaultOptionKeys.in_features: {
            "explanation": "Single source timestamp column to sessionize",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
        PARTITION_BY: {
            "explanation": "List of columns to partition by (default: whole table as one stream)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
        ORDER_BY: {
            "explanation": "Column to order by (ascending); defaults to the source timestamp column",
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
        source_names = [str(f.name) for f in in_features_set]
        if len(source_names) > self.MAX_IN_FEATURES:
            raise ValueError(
                f"sessionize supports at most {self.MAX_IN_FEATURES} source feature, "
                f"but got {len(source_names)}: {source_names}"
            )
        return set(in_features_set)

    @classmethod
    def _extract_source_features(cls, feature: Feature) -> list[str]:
        """Extract and validate the single source feature.

        Returns a one-element list containing the source column name. Raises
        ``ValueError`` if more than one source feature is found, since
        sessionize supports at most one source column.
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
                f"sessionize requires at least {cls.MIN_IN_FEATURES} source feature, "
                f"but got {len(source_names)} (in_features is empty)."
            )

        if len(source_names) > cls.MAX_IN_FEATURES:
            raise ValueError(
                f"sessionize supports at most {cls.MAX_IN_FEATURES} source feature, "
                f"but got {len(source_names)}: {source_names}"
            )

        return source_names

    @classmethod
    def _extract_threshold_token(cls, feature: Feature) -> str:
        """Return the ``sessionize_{n}_{unit}`` token from the feature name."""
        name = feature.name
        try:
            token = name.split("__sessionize_", 1)[1]
        except IndexError as exc:
            raise ValueError(f"Could not extract a sessionize token from feature name {name!r}.") from exc
        return f"sessionize_{token}"

    @classmethod
    def _extract_partition_by(cls, feature: Feature) -> list[str]:
        """Return ``partition_by`` as a list (defaulting to ``[]`` when absent)."""
        partition_by = feature.options.get(cls.PARTITION_BY)
        if partition_by is None:
            return []
        return list(partition_by)

    @classmethod
    def _extract_order_by(cls, feature: Feature, source_col: str) -> str:
        """Return ``order_by``, defaulting to the source timestamp column when absent."""
        order_by = feature.options.get(cls.ORDER_BY)
        if order_by is None:
            return source_col
        return str(order_by)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Compute one session-id column per feature in ``features``."""
        table = data

        for feature in features.features:
            feature_name = feature.name

            source_features = cls._extract_source_features(feature)
            source_col = source_features[0]
            token = cls._extract_threshold_token(feature)
            n, unit = _parse_sessionize_op(token)
            threshold_seconds = _sessionize_threshold_seconds(n, unit)
            partition_by = cls._extract_partition_by(feature)
            order_by = cls._extract_order_by(feature, source_col)

            cls._assert_source_column_present(table, order_by)

            table = cls._compute_session(table, feature_name, order_by, threshold_seconds, partition_by)

        return table

    @classmethod
    def _assert_source_column_present(cls, data: Any, order_col: str) -> None:
        """Reject a missing source column with a clear ``ValueError`` (backend-specific)."""
        raise NotImplementedError

    @classmethod
    def _compute_session(
        cls,
        data: Any,
        feature_name: str,
        order_col: str,
        threshold_seconds: int,
        partition_by: list[str],
    ) -> Any:
        raise NotImplementedError
