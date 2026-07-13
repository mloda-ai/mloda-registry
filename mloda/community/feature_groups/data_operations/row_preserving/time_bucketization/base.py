"""Base class for time bucketization feature groups.

Buckets a timestamp column to a coarser interval via ``floor`` / ``ceil`` /
``round``. Supports DuckDB, SQLite, Pandas, Polars, and PyArrow backends.

Pattern: ``{src}__{op}_{n}_{unit}``

Examples::

    "timestamp__floor_1_day"       # day floor (midnight UTC)
    "timestamp__ceil_5_minute"     # next 5-minute boundary
    "timestamp__round_1_hour"      # nearest hour (half rounds up)
    "timestamp__floor_1_week"      # ISO-Monday week start

Supported units are ``minute``, ``hour``, ``day``, ``week``, ``month``,
``year``. The bucket size ``n`` is a positive integer; only ``n=1`` is
supported for ``week`` / ``month`` / ``year`` in v1.

Semantics pinned across all backends:

- **Week start**: ISO Monday (Sun 2023-01-01 floors to Mon 2022-12-26).
- **Round tie-break**: half-up (every midpoint rounds toward the next
  bucket), matching PyArrow's ``round_temporal`` and Polars' ``dt.round``.
- **Idempotency**: ``ceil(aligned, X) == aligned`` for fixed-freq units
  (``minute`` / ``hour`` / ``day``). Calendar units (``week`` / ``month``
  / ``year``) always advance to the next bucket on aligned input, matching
  PyArrow's ``ceil_temporal`` behaviour.
- **Timezone**: preserved from input.
- **Null**: propagated.
- **Output dtype**: same timestamp type as input.

The ``bucket_op`` option (full op token, e.g. ``"floor_1_day"``) mirrors
datetime FG's ``datetime_op``.
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

from mloda.community.feature_groups.data_operations.base import is_op_token

TIME_BUCKETIZATION_OPS: dict[str, str] = {
    "floor": "Round timestamp down to the start of the enclosing bucket",
    "ceil": "Round timestamp up to the start of the next bucket (idempotent on aligned for fixed-freq units; always advances for week/month/year)",
    "round": "Round timestamp to the nearest bucket boundary (half rounds up)",
}

TIME_BUCKETIZATION_UNITS: dict[str, str] = {
    "minute": "Minute-aligned buckets (sub-day, fixed length)",
    "hour": "Hour-aligned buckets (sub-day, fixed length)",
    "day": "Day-aligned buckets (calendar day, midnight UTC)",
    "week": "ISO-Monday-anchored week buckets (n=1 only)",
    "month": "Calendar-month buckets (n=1 only, non-uniform length)",
    "year": "Calendar-year buckets (n=1 only, non-uniform length)",
}

# Calendar units only support ``n=1`` in v1.
_CALENDAR_UNITS: frozenset[str] = frozenset({"week", "month", "year"})


def _parse_bucket_op(token: str) -> tuple[str, int, str]:
    """Parse a bucket-op token into ``(op, n, unit)``.

    The token format is ``{op}_{n}_{unit}`` with three underscore-separated
    parts (e.g. ``"floor_1_day"``, ``"ceil_15_minute"``).

    Raises:
        ValueError: if the token is malformed, the op or unit is unknown,
            ``n`` is not a positive integer, or ``n > 1`` is requested for
            a calendar unit (``week`` / ``month`` / ``year``).
    """
    parts = token.split("_")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid bucket_op token {token!r}: expected '{{op}}_{{n}}_{{unit}}', "
            f"got {len(parts)} underscore-separated parts."
        )

    op, n_str, unit = parts

    if op not in TIME_BUCKETIZATION_OPS:
        raise ValueError(f"Unsupported bucket op {op!r} in {token!r}; supported: {sorted(TIME_BUCKETIZATION_OPS)}.")

    if unit not in TIME_BUCKETIZATION_UNITS:
        raise ValueError(
            f"Unsupported time bucketization unit {unit!r} in {token!r}; supported: {sorted(TIME_BUCKETIZATION_UNITS)}."
        )

    try:
        n = int(n_str)
    except ValueError as exc:
        raise ValueError(f"Bucket size in {token!r} must be a positive integer, got {n_str!r}.") from exc

    if n <= 0:
        raise ValueError(f"Bucket size n must be a positive integer (n > 0), got {n} in {token!r}.")

    if unit in _CALENDAR_UNITS and n != 1:
        raise ValueError(f"Only n=1 is supported for calendar unit {unit!r} in v1; got n={n} in {token!r}.")

    return op, n, unit


class TimeBucketizationFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Base class for element-wise timestamp bucketization.

    Subclasses must implement ``_compute_bucket`` (the backend-specific
    arithmetic) and ``_assert_source_column_is_timestamp`` (the dtype guard).
    """

    # Regex captures the FULL op token (e.g. ``floor_1_day``) in a single
    # group. The validation in ``_validate_string_match`` calls
    # ``_parse_bucket_op`` to reject ``n=0`` and ``n>1`` calendar-unit tokens.
    PREFIX_PATTERN = r".*__((?:floor|ceil|round)_\d+_(?:minute|hour|day|week|month|year))$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    BUCKET_OP = "bucket_op"

    @staticmethod
    def _is_valid_bucket_op_value(value: Any) -> bool:
        """Validation hook for the open-ended ``bucket_op`` token domain.

        Tokens follow ``{op}_{n}_{unit}`` with too many valid combinations
        to enumerate in ``PROPERTY_MAPPING``. This validator delegates to
        ``_parse_bucket_op`` (returning False on ValueError) so that the
        config-based ``match_feature_group_criteria`` accepts well-formed
        tokens and rejects malformed / unsupported ones.
        """
        if not isinstance(value, str):
            return False
        try:
            _parse_bucket_op(value)
        except ValueError:
            return False
        return True

    PROPERTY_MAPPING = {
        BUCKET_OP: {
            "explanation": "Full bucketization op token (e.g. 'floor_1_day', 'ceil_5_minute')",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.element_validator: _is_valid_bucket_op_value,
            DefaultOptionKeys.match_guard: is_op_token,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Single source timestamp column to bucketize",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
    }

    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, source_feature: str) -> bool:
        """Reject pattern matches whose token fails ``_parse_bucket_op`` validation.

        The regex accepts any ``\\d+`` for ``n``, so ``floor_0_day`` and
        ``floor_2_week`` match grammatically. This method enforces the n=0
        and n>1-calendar-unit constraints by parsing and returning False on
        ValueError (rather than letting it propagate, so match resolution
        can try other feature groups cleanly).
        """
        try:
            _parse_bucket_op(operation_config)
        except ValueError:
            return False
        return True

    @classmethod
    def get_bucket_op(cls, feature_name: str) -> str:
        """Extract the full op token from a string-pattern feature name."""
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        raise ValueError(f"Could not extract bucket op from feature name: {feature_name}")

    @classmethod
    def _extract_bucket_op(cls, feature: Feature) -> str:
        """Extract the op token from the feature name or from Options."""
        feature_name = feature.name
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        op = feature.options.get(cls.BUCKET_OP)
        if op is None:
            raise ValueError(f"Could not extract bucket op for {feature_name}")
        return str(op)

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
        """Extract and validate the single source feature.

        Returns a one-element list containing the source column name.
        Raises ValueError if more than one source feature is found, since
        time bucketization only supports a single source column.
        """
        feature_name = feature.name
        prefix_patterns = cls._get_prefix_patterns()

        operation_config, source_feature = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)

        if operation_config and source_feature:
            return [source_feature]

        in_features_set = feature.options.get_in_features()
        source_names: list[str] = [str(f.name) for f in in_features_set]

        if len(source_names) < cls.MIN_IN_FEATURES:
            raise ValueError(
                f"Time bucketization requires at least {cls.MIN_IN_FEATURES} source feature, "
                f"but got {len(source_names)} (in_features is empty)."
            )

        if len(source_names) > cls.MAX_IN_FEATURES:
            raise ValueError(
                f"Time bucketization supports at most {cls.MAX_IN_FEATURES} source feature, "
                f"but got {len(source_names)}: {source_names}"
            )

        return source_names

    @staticmethod
    def _raise_non_timestamp_source(source_col: str, got: object) -> None:
        """Shared error format for the timestamp-source contract.

        Backend overrides of ``_assert_source_column_is_timestamp`` call
        this helper so the message stays uniform across all backends.
        """
        raise ValueError(
            f"Source column {source_col!r} must be a timestamp/datetime for time bucketization; got {got}."
        )

    @classmethod
    def _assert_source_column_is_timestamp(cls, data: Any, source_col: str) -> None:
        """Reject non-timestamp source columns with a clear ``ValueError``.

        Backend-specific; implemented per backend. Implementations should
        call ``cls._raise_non_timestamp_source(source_col, <native dtype>)``
        to keep the message format uniform.
        """
        raise NotImplementedError

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Compute one bucketization column per feature in ``features``.

        For each feature:

        1. Extract the source feature name (string-pattern or Options).
        2. Extract the op token (string-pattern or Options).
        3. Parse the op token via ``_parse_bucket_op`` (raises ValueError on
           invalid tokens such as ``n=0`` or ``floor_2_week``).
        4. Assert the source column is a timestamp/datetime type.
        5. Dispatch to ``_compute_bucket`` for the backend-specific math.

        Null timestamps propagate to null output. The output column has the
        same timestamp type (resolution, tz) as the input.
        """
        table = data

        for feature in features.features:
            feature_name = feature.name

            source_features = cls._extract_source_features(feature)
            source_col = source_features[0]
            op_token = cls._extract_bucket_op(feature)

            op, n, unit = _parse_bucket_op(op_token)

            cls._assert_source_column_is_timestamp(table, source_col)

            table = cls._compute_bucket(table, feature_name, source_col, op, n, unit)

        return table

    @classmethod
    def _compute_bucket(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        op: str,
        n: int,
        unit: str,
    ) -> Any:
        raise NotImplementedError
