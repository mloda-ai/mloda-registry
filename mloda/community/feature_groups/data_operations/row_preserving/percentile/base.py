"""Base class for percentile feature groups."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mloda.provider import (
    DefaultOptionKeys,
    FeatureChainParserMixin,
    FeatureSet,
    OperationFeatureGroup,
    property_spec,
)
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.base import (
    SCALAR_NUMBER_EXPECTED,
    KeyColumnInputsMixin,
    assert_key_columns_present,
    assert_source_columns_present,
    is_scalar_number,
    scalar_number_value,
)
from mloda.community.feature_groups.data_operations.mask_utils import MASK_KEY, parse_mask_spec


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


def _percent_token_to_unit(token: str) -> float | None:
    """The 0.0-1.0 fraction of a ``pN`` name token with N in 0-100, else None."""
    n = int(token[1:])
    if 0 <= n <= 100:
        return n / 100.0
    return None


def _is_unit_interval_element(value: object) -> bool:
    """Bare element predicate: one number in [0.0, 1.0], compared without float conversion so a huge int cannot overflow."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= value <= 1.0


class PercentileFeatureGroup(KeyColumnInputsMixin, FeatureChainParserMixin, OperationFeatureGroup):
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

    KEY_COLUMN_OPTIONS = ("partition_by", "mask")

    PERCENTILE = "percentile"
    PARTITION_BY = "partition_by"

    PREFIX_PATTERN = rf".*__(?P<{PERCENTILE}>p\d+)_percentile$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    PROPERTY_MAPPING = {
        PERCENTILE: property_spec(
            "Percentile value (float between 0.0 and 1.0)",
            strict=True,
            element_validator=_is_unit_interval_element,
            match_guard=is_scalar_number,
            expected=SCALAR_NUMBER_EXPECTED,
            name_converter=_percent_token_to_unit,
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
    def match_feature_group_criteria(
        cls,
        feature_name: Any,
        options: Any,
        _data_access_collection: Any = None,
    ) -> bool:
        """Extend mixin matching with partition_by validation (must be a non-empty list of strings)."""
        if not super().match_feature_group_criteria(feature_name, options, _data_access_collection):
            return False

        partition_by = options.get(cls.PARTITION_BY)
        if not isinstance(partition_by, (list, tuple)):
            return False
        if len(partition_by) == 0:
            return False
        if not all(isinstance(item, str) for item in partition_by):
            return False

        return True

    @classmethod
    def get_percentile_value(cls, feature_name: str) -> float:
        """Extract percentile float from a feature name string.

        Parses the ``pN`` portion from a feature name matching PREFIX_PATTERN
        and returns ``N / 100.0``.

        Raises ValueError if the feature name does not match.
        """
        result = cls.resolve_feature_name(feature_name).bindings.get(cls.PERCENTILE)
        if result is not None:
            return float(result)
        raise ValueError(f"Could not extract percentile value from feature name: {feature_name}")

    @classmethod
    def _extract_percentile(cls, feature: Feature) -> float:
        """Extract percentile float from feature (name first, then options)."""
        feature_name = feature.name
        resolution = cls.resolve_feature_name(feature_name)
        result = resolution.bindings.get(cls.PERCENTILE)
        if result is not None:
            return float(result)
        if resolution.owned:
            raise ValueError(f"Could not extract percentile for {feature_name}")
        percentile = _unit_percentile(feature.options.get(cls.PERCENTILE))
        if percentile is None:
            raise ValueError(f"Could not extract percentile for {feature_name}")
        return percentile

    @classmethod
    def percentile(
        cls,
        data: Any,
        features: FeatureSet,
        *,
        compute_percentile: Callable[[Any, str, str, list[str], float, list[tuple[str, str, Any]] | None], Any],
    ) -> Any:
        """Compute a percentile per source column, partitioned, and broadcast to all rows.

        Each feature in the feature set produces one new column containing the
        percentile value repeated for every row in the partition.
        """
        table = data

        for feature in features.features:
            feature_name = feature.name

            source_col = cls._extract_single_source_feature(feature)
            assert_source_columns_present(data, [source_col])
            percentile = cls._extract_percentile(feature)
            partition_by = feature.options.get(cls.PARTITION_BY)
            if not isinstance(partition_by, (list, tuple)) or not partition_by:
                raise ValueError(
                    f"percentile requires a non-empty partition_by, got {partition_by!r} for feature {feature_name!r}."
                )
            partition_by = list(partition_by)
            mask_spec = parse_mask_spec(feature.options.get(MASK_KEY))
            assert_key_columns_present(data, partition_by, mask_spec=mask_spec)

            table = compute_percentile(table, feature_name, source_col, partition_by, percentile, mask_spec)

        return table
