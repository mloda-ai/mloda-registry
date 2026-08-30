"""Base class for percentile feature groups."""

from __future__ import annotations

import logging
import os
from typing import Any

from mloda.core.abstract_plugins.components.utils import escalate_match_abort  # no public equivalent yet
from mloda.provider import DefaultOptionKeys, FeatureChainParser, FeatureGroup, FeatureSet, property_spec
from mloda.user import Feature, FeatureName, Options

from mloda.community.feature_groups.data_operations.base import (
    RejectionReasonMixin,
    is_scalar_number,
    scalar_number_value,
)
from mloda.community.feature_groups.data_operations.mask_utils import MASK_KEY, parse_mask_spec

logger = logging.getLogger(__name__)


def _unit_percentile(value: object) -> float | None:
    """The option as a float in 0.0-1.0, or None when it is not one number in that range.

    The range check runs BEFORE the conversion on purpose: an int-to-float comparison is exact
    at any magnitude, while ``float(10**400)`` raises OverflowError, which discovery does not
    catch and which a user-supplied option must never trigger.
    """
    if not is_scalar_number(value):
        return None
    number = scalar_number_value(value)
    if number < 0.0 or number > 1.0:
        return None
    return float(number)


def _is_unit_interval_element(value: object) -> bool:
    """Bare element predicate: one number in [0.0, 1.0], compared without float conversion so a huge int cannot overflow."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= value <= 1.0


class PercentileFeatureGroup(RejectionReasonMixin, FeatureGroup):
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
        # deferred_binding stays True: see _validate_forwarded_percentile_mismatch for why.
        PERCENTILE: property_spec(
            "Percentile value (float between 0.0 and 1.0)",
            strict=True,
            element_validator=_is_unit_interval_element,
            match_guard=is_scalar_number,
            deferred_binding=True,
        ),
        DefaultOptionKeys.in_features: property_spec(
            "Source feature for percentile computation",
            strict=False,
        ),
        PARTITION_BY: property_spec(
            "List of columns to partition by",
            strict=False,
        ),
        MASK_KEY: property_spec(
            "Conditional mask: (column, operator, value) tuple or list of tuples",
            strict=False,
            default=None,
        ),
    }

    @classmethod
    def _parse_percentile_from_config(cls, operation_config: str) -> float | None:
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
        """Extend mixin matching with forwarded-percentile protection and partition_by validation.

        The mixin handles:
        - Pattern and config matching via PROPERTY_MAPPING
        - List-valued options (partition_by) via tuple conversion
        - MIN/MAX_IN_FEATURES enforcement

        We add:
        - forwarded-value vs. name-parsed-value protection for ``percentile``
        - percentile range validation for config-based features (0.0-1.0)
        - partition_by type validation (must be a list of strings)
        """
        if not super().match_feature_group_criteria(feature_name, options, _data_access_collection):
            return False

        # None covers both halves: unresolvable, and resolved but outside 0.0-1.0.
        resolved_percentile, name_matched = cls._resolve_percentile(feature_name, options)
        if resolved_percentile is None:
            return False

        if name_matched:
            cls._validate_forwarded_percentile_mismatch(feature_name, options, resolved_percentile)

        partition_by = options.get(cls.PARTITION_BY)
        if not isinstance(partition_by, (list, tuple)):
            return False
        if len(partition_by) == 0:
            return False
        if not all(isinstance(item, str) for item in partition_by):
            return False

        return True

    @classmethod
    def _validate_forwarded_percentile_mismatch(
        cls, feature_name: Any, options: Any, resolved_percentile: float
    ) -> None:
        """Reject a forwarded ``percentile`` that contradicts the name-parsed value.

        PREFIX_PATTERN captures a "pN" percent token, while ``percentile`` is a 0.0-1.0 fraction, so
        the two can't share a name binding the way bucket_op / rank_type / offset_type do; this
        compares the shared float value directly instead.
        """
        if cls.PERCENTILE not in (options.inherited_group_keys or ()):
            return
        forwarded = options.get(cls.PERCENTILE)
        forwarded_value = _unit_percentile(forwarded)
        if forwarded_value is None or forwarded_value == resolved_percentile:
            return
        message = (
            f"Feature '{feature_name}': option '{cls.PERCENTILE}' was forwarded from a consumer with value "
            f"'{forwarded}', but the feature name parses to percentile {resolved_percentile}. The name-parsed "
            f"value takes precedence, so the forwarded value would be silently ignored. Carve the key out with "
            f"forward_group_exclude={{'{cls.PERCENTILE}'}} on the child in the consumer's input_features, or use "
            f"an allowlist / forward_group=False. Set MLODA_ALLOW_FORWARDED_NAME_MISMATCH=1 to downgrade this "
            f"error to a warning."
        )
        if os.environ.get("MLODA_ALLOW_FORWARDED_NAME_MISMATCH", "").lower() in ("1", "true"):
            logger.warning(message)
            return
        raise escalate_match_abort(ValueError(message))

    @classmethod
    def _resolve_percentile(cls, feature_name: Any, options: Any) -> tuple[float | None, bool]:
        """Extract percentile as a float in 0.0-1.0 from feature name or options; None if there is none.

        Second element is True when the name identified the group.
        """
        name = str(feature_name)
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(name, prefix_patterns)
        if operation_config is not None:
            return cls._parse_percentile_from_config(operation_config), True
        return _unit_percentile(options.get(cls.PERCENTILE)), False

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
    def _extract_percentile(cls, feature: Feature) -> float:
        """Extract percentile float from feature (name first, then options)."""
        feature_name = feature.name
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            result = cls._parse_percentile_from_config(operation_config)
            if result is not None:
                return result
        percentile = _unit_percentile(feature.options.get(cls.PERCENTILE))
        if percentile is None:
            raise ValueError(f"Could not extract percentile for {feature_name}")
        return percentile

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        """Parse input features from feature name or options."""
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
        """Extract and validate the single source feature for percentile.

        Returns a one-element list containing the source column name.
        Raises ValueError if more than one source feature is found, since
        this package only supports single-column percentile computation.
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
            feature_name = feature.name

            source_features = cls._extract_source_features(feature)
            source_col = source_features[0]
            percentile = cls._extract_percentile(feature)
            partition_by = feature.options.get(cls.PARTITION_BY)
            if not isinstance(partition_by, (list, tuple)) or not partition_by:
                raise ValueError(
                    f"percentile requires a non-empty partition_by, got {partition_by!r} for feature {feature_name!r}."
                )
            partition_by = list(partition_by)
            mask_spec = parse_mask_spec(feature.options.get(MASK_KEY))

            table = cls._compute_percentile(table, feature_name, source_col, partition_by, percentile, mask_spec)

        return table

    @classmethod
    def _compute_percentile(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        percentile: float,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> Any:
        """Subclasses must implement the actual percentile computation."""
        raise NotImplementedError
