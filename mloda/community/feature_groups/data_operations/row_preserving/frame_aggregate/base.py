"""Base class for frame aggregate feature groups."""

from __future__ import annotations

import functools
import re
from typing import Any

from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import DefaultOptionKeys, FeatureGroup

from mloda.community.feature_groups.data_operations.base import (
    FRAME_SIZE as _FRAME_SIZE_KEY,
    FRAME_TYPE as _FRAME_TYPE_KEY,
    FRAME_UNIT as _FRAME_UNIT_KEY,
    RejectionReasonMixin,
    always_required,
    column_ref_value,
    is_column_ref,
    is_in_features_value,
    is_op_token,
    is_positive_int,
    op_token_value,
    option_value,
    positive_int_value,
)
from mloda.community.feature_groups.data_operations.capability_hook import SubtypeCapabilityHook
from mloda.community.feature_groups.data_operations.mask_utils import MASK_KEY, parse_mask_spec


# Patterns for string-based feature names; group 1 is the aggregation token, matching what
# core's FeatureChainParser.parse_feature_name returns as the operation:
#   {col}__sum_rolling_3         -> rolling N rows
#   {col}__avg_5_day_window      -> time-interval window
#   {col}__cumsum                -> cumulative sum
#   {col}__expanding_avg         -> expanding window
# The size group is [1-9]\d*, not \d+: a zero-sized or zero-padded frame is not a window, and the
# string path must reject it in the pattern itself, exactly as the config path rejects frame_size=0.
_ROLLING_PATTERN = re.compile(r"^.+__(\w+)_rolling_([1-9]\d*)$")
_TIME_WINDOW_PATTERN = re.compile(r"^.+__(\w+)_([1-9]\d*)_(\w+)_window$")
_CUMULATIVE_PATTERN = re.compile(r"^.+__cum(\w+)$")
_EXPANDING_PATTERN = re.compile(r"^.+__expanding_(\w+)$")

# Frame aggregation supports the order-independent subset of the canonical
# aggregation-type table (no mode/nunique/first/last, no ddof-variant spellings).
# Frame never uses the canonical descriptions, so this is an explicit set rather
# than a derivation from the shared table. Cumulative and expanding accept this
# full set, so they need no narrower table of their own.
_AGGREGATION_TYPES = frozenset({"sum", "avg", "count", "min", "max", "std", "var", "median"})
_TIME_UNITS = {"second", "minute", "hour", "day", "week", "month", "year"}


def _option_token(options: Options, key: str) -> str | None:
    """An option as a bare token, unwrapped from its container; None when the option is absent."""
    return option_value(options, key, op_token_value)


def _needs_frame_size(options: Options) -> bool:
    """Rolling and time frames are sized; cumulative and expanding are not."""
    return _option_token(options, _FRAME_TYPE_KEY) in ("rolling", "time")


def _needs_frame_unit(options: Options) -> bool:
    """Only a time-interval frame carries a unit."""
    return _option_token(options, _FRAME_TYPE_KEY) == "time"


@functools.lru_cache(maxsize=1024)
def _parse_frame_feature_cached(feature_name: str) -> dict[str, Any] | None:
    """Regex-parse a frame feature name; cached on the name since the regexes are module constants."""
    # Same split core uses, so the parsed source column cannot drift from the routed one.
    source_col = feature_name.rsplit("__", 1)[0]
    if not source_col:
        return None

    m = _ROLLING_PATTERN.match(feature_name)
    if m:
        return {
            "source_col": source_col,
            "agg_type": m.group(1),
            "frame_type": "rolling",
            "frame_size": int(m.group(2)),
            "frame_unit": None,
        }

    m = _TIME_WINDOW_PATTERN.match(feature_name)
    if m:
        return {
            "source_col": source_col,
            "agg_type": m.group(1),
            "frame_type": "time",
            "frame_size": int(m.group(2)),
            "frame_unit": m.group(3),
        }

    m = _CUMULATIVE_PATTERN.match(feature_name)
    if m:
        return {
            "source_col": source_col,
            "agg_type": m.group(1),
            "frame_type": "cumulative",
            "frame_size": None,
            "frame_unit": None,
        }

    m = _EXPANDING_PATTERN.match(feature_name)
    if m:
        return {
            "source_col": source_col,
            "agg_type": m.group(1),
            "frame_type": "expanding",
            "frame_size": None,
            "frame_unit": None,
        }

    return None


class FrameAggregateFeatureGroup(SubtypeCapabilityHook, RejectionReasonMixin, FeatureGroup):
    """Base class for frame aggregate operations that preserve row count.

    Frame aggregation computes an aggregate over a sliding or expanding window
    within partitioned, ordered groups. The output always has the same number
    of rows as the input.

    ## Supported Frame Types

    - ``rolling``: Fixed-size row-count window (last N rows).
    - ``time``: Time-interval window (last N days/hours/etc.).
    - ``cumulative``: Running aggregate from the first row to the current row.
    - ``expanding``: Same as cumulative (alias for clarity).

    Subclasses declare which frame types they support via
    ``SUPPORTED_FRAME_TYPES``. Features requesting an unsupported frame type
    are rejected at discovery time (match_feature_group_criteria returns False).

    ## Supported Aggregation Types

    - ``sum``, ``avg``, ``count``, ``min``, ``max``
    - ``std``, ``var``, ``median`` (not all frameworks support all)

    ## Feature Creation Methods

    ### 1. String-Based Creation

    Features follow one of four naming patterns:

    - Rolling: ``{col}__{agg}_rolling_{N}`` (e.g. ``sales__sum_rolling_3``)
    - Time window: ``{col}__{agg}_{size}_{unit}_window`` (e.g. ``sales__avg_7_day_window``)
    - Cumulative: ``{col}__cum{agg}`` (e.g. ``sales__cumsum``)
    - Expanding: ``{col}__expanding_{agg}`` (e.g. ``sales__expanding_avg``)

    All require ``partition_by`` and ``order_by`` in Options context.

    ### 2. Configuration-Based Creation

    Uses Options with proper context parameter separation::

        feature = Feature(
            name="my_result",
            options=Options(
                context={
                    "aggregation_type": "sum",
                    "frame_type": "rolling",
                    "frame_size": 3,
                    "in_features": "sales",
                    "partition_by": ["region"],
                    "order_by": "timestamp",
                }
            ),
        )

    ## Parameter Classification

    ### Context Parameters
    - ``aggregation_type``: The aggregation function to apply
    - ``frame_type``: One of rolling, time, cumulative, expanding
    - ``frame_size``: Window size (rows for rolling, integer for time)
    - ``frame_unit``: Time unit (day, hour, etc.) for time windows
    - ``in_features``: The source feature to aggregate
    - ``partition_by``: List of columns to partition by
    - ``order_by``: Column to order by (required for all frame types)
    """

    # PREFIX_PATTERN is the rolling member; FRAME_PATTERNS is the full matching set the mixin uses.
    FRAME_PATTERNS: tuple[str, ...] = (
        _ROLLING_PATTERN.pattern,
        _TIME_WINDOW_PATTERN.pattern,
        _CUMULATIVE_PATTERN.pattern,
        _EXPANDING_PATTERN.pattern,
    )
    PREFIX_PATTERN = _ROLLING_PATTERN.pattern

    SUPPORTED_FRAME_TYPES: set[str] = {"rolling", "time", "cumulative", "expanding"}
    SUPPORTED_TIME_UNITS: set[str] = _TIME_UNITS

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    #: agg-type support depends on frame_type.
    _CAPABILITY_HAS_AXIS: bool = True

    AGGREGATION_TYPE = "aggregation_type"
    # Aliases of the shared keys, so the literals stay defined once in data_operations/base.py.
    FRAME_TYPE = _FRAME_TYPE_KEY
    FRAME_SIZE = _FRAME_SIZE_KEY
    FRAME_UNIT = _FRAME_UNIT_KEY
    PARTITION_BY = "partition_by"
    ORDER_BY = "order_by"

    #: Keys a frame name supplies via _parse_frame_feature; exempt from required_when on the name path.
    _NAME_SUPPLIED_KEYS: tuple[str, ...] = (FRAME_TYPE, FRAME_SIZE, FRAME_UNIT)

    PROPERTY_MAPPING = {
        AGGREGATION_TYPE: {
            "explanation": "Aggregation applied over the frame",
            DefaultOptionKeys.allowed_values: {k: k for k in _AGGREGATION_TYPES},
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.match_guard: is_op_token,
        },
        FRAME_TYPE: {
            "explanation": "Frame semantics of the window",
            DefaultOptionKeys.allowed_values: {
                "rolling": "Fixed-size row-count window",
                "time": "Time-interval window",
                "cumulative": "Running aggregate from start",
                "expanding": "Same as cumulative",
            },
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.match_guard: is_op_token,
        },
        FRAME_SIZE: {
            "explanation": "Window size (rows for rolling, integer for time)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.match_guard: is_positive_int,
            DefaultOptionKeys.required_when: _needs_frame_size,
        },
        FRAME_UNIT: {
            "explanation": "Time unit for time windows",
            DefaultOptionKeys.allowed_values: {unit: f"{unit} interval" for unit in sorted(_TIME_UNITS)},
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.match_guard: is_op_token,
            DefaultOptionKeys.required_when: _needs_frame_unit,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature for frame aggregation",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.match_guard: is_in_features_value,
        },
        PARTITION_BY: {
            "explanation": "List of columns to partition by",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        },
        ORDER_BY: {
            "explanation": "Column to order by within each partition",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.match_guard: is_column_ref,
            DefaultOptionKeys.required_when: always_required,
        },
        MASK_KEY: {
            "explanation": "Conditional mask: (column, operator, value) tuple or list of tuples",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.default: None,
        },
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        """Parse input features from the four frame patterns or config fallback."""
        name = str(feature_name)
        parsed = self._parse_frame_feature(name)
        if parsed is not None:
            return {Feature(parsed["source_col"])}
        in_features_set = options.get_in_features()
        return set(in_features_set)

    @classmethod
    def _extract_source_features(cls, feature: Feature) -> list[str]:
        """Extract source features from the four frame patterns or config fallback."""
        name = feature.name
        parsed = cls._parse_frame_feature(name)
        if parsed is not None:
            return [parsed["source_col"]]
        in_features_set = feature.options.get_in_features()
        return [str(f.name) for f in in_features_set]

    @classmethod
    def _parse_frame_feature(cls, feature_name: str) -> dict[str, Any] | None:
        """Parse a frame aggregate feature name into its components.

        Returns a dict with keys: source_col, agg_type, frame_type, frame_size, frame_unit.
        Returns None if the name doesn't match any pattern.
        """
        # Copy: the cache shares one dict across callers, so mutations must not leak.
        cached = _parse_frame_feature_cached(feature_name)
        return None if cached is None else dict(cached)

    @classmethod
    def _get_prefix_patterns(cls) -> list[str]:
        """All four name shapes, so the mixin routes every frame pattern, not only rolling."""
        return list(cls.FRAME_PATTERNS)

    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, source_feature: str) -> bool:
        """Name path: the parsed agg type, frame type and time unit must all be supported here."""
        parsed = cls._parse_frame_feature(str(feature_name))
        if parsed is None:
            return False
        if parsed["agg_type"] not in _AGGREGATION_TYPES:
            return False
        if parsed["frame_type"] not in cls.SUPPORTED_FRAME_TYPES:
            return False
        # SUPPORTED_TIME_UNITS defaults to _TIME_UNITS and subclasses only narrow it, so it subsumes it.
        if parsed["frame_type"] == "time" and parsed["frame_unit"] not in cls.SUPPORTED_TIME_UNITS:
            return False
        return True

    @classmethod
    def _validate_required_when(
        cls,
        result: bool,
        feature_name: str | FeatureName,
        prefix_patterns: list[str],
        property_mapping: dict[str, Any] | None,
        options: Options,
    ) -> bool:
        # A frame name carries its own type, size and unit; order_by stays required on every path.
        if property_mapping is not None and cls._parse_frame_feature(str(feature_name)) is not None:
            property_mapping = {k: v for k, v in property_mapping.items() if k not in cls._NAME_SUPPLIED_KEYS}
        return super()._validate_required_when(result, feature_name, prefix_patterns, property_mapping, options)

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Any,
        options: Any,
        _data_access_collection: Any = None,
    ) -> bool:
        """Extend the declaration-driven mixin match with per-subclass capability and shape checks.

        PROPERTY_MAPPING owns the value spaces; order_by is declared always-required, frame_size and
        frame_unit stay config-path only (the name supplies them), and partition_by presence is hand-enforced below.
        """
        if not super().match_feature_group_criteria(feature_name, options, _data_access_collection):
            return False

        # Config path only: SUPPORTED_* narrows per subclass, so it cannot be declared once on the base.
        if cls._parse_frame_feature(str(feature_name)) is None:
            frame_type = _option_token(options, cls.FRAME_TYPE)
            if frame_type not in cls.SUPPORTED_FRAME_TYPES:
                return False
            if frame_type == "time" and _option_token(options, cls.FRAME_UNIT) not in cls.SUPPORTED_TIME_UNITS:
                return False

        partition_by = options.get(cls.PARTITION_BY)
        if not isinstance(partition_by, (list, tuple)):
            return False
        if not all(isinstance(item, str) for item in partition_by):
            return False

        return True

    @classmethod
    def _capability_subtype(cls, feature_name: str, options: Options) -> str | None:
        parsed = cls._parse_frame_feature(feature_name)
        if parsed is not None:
            return str(parsed["agg_type"])
        agg_type = options.get(cls.AGGREGATION_TYPE)
        return None if agg_type is None else op_token_value(agg_type)

    @classmethod
    def _capability_secondary(cls, feature_name: str, options: Options) -> str | None:
        parsed = cls._parse_frame_feature(feature_name)
        if parsed is not None:
            return str(parsed["frame_type"])
        frame_type = options.get(cls.FRAME_TYPE)
        return None if frame_type is None else op_token_value(frame_type)

    @classmethod
    def _capability_guard(cls, feature_name: str, options: Options) -> bool:
        """Reject frame types and time units the backend cannot compute; unresolved frame type stays True."""
        frame_type = cls._capability_secondary(feature_name, options)
        if frame_type is None:
            return True
        if frame_type not in cls.SUPPORTED_FRAME_TYPES:
            return False
        if frame_type == "time":
            parsed = cls._parse_frame_feature(feature_name)
            frame_unit = parsed["frame_unit"] if parsed is not None else options.get(cls.FRAME_UNIT)
            if frame_unit is not None and op_token_value(frame_unit) not in cls.SUPPORTED_TIME_UNITS:
                return False
        return True

    @classmethod
    def _extract_params(cls, feature: Feature) -> dict[str, Any]:
        """Extract all frame aggregate parameters from a feature."""
        feature_name = feature.name
        parsed = cls._parse_frame_feature(feature_name)

        if parsed is not None:
            return {
                "source_col": parsed["source_col"],
                "agg_type": parsed["agg_type"],
                "frame_type": parsed["frame_type"],
                "frame_size": parsed["frame_size"],
                "frame_unit": parsed["frame_unit"],
                "partition_by": feature.options.get(cls.PARTITION_BY),
                "order_by": option_value(feature.options, cls.ORDER_BY, column_ref_value),
            }

        source_features = cls._extract_source_features(feature)
        return {
            "source_col": source_features[0],
            "agg_type": op_token_value(feature.options.get(cls.AGGREGATION_TYPE)),
            "frame_type": op_token_value(feature.options.get(cls.FRAME_TYPE)),
            "frame_size": option_value(feature.options, cls.FRAME_SIZE, positive_int_value),
            "frame_unit": _option_token(feature.options, cls.FRAME_UNIT),
            "partition_by": feature.options.get(cls.PARTITION_BY),
            "order_by": option_value(feature.options, cls.ORDER_BY, column_ref_value),
        }

    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> DataType | None:
        """Declare INT64 for count (a counting op); other aggregates stay open."""
        agg_type = cls._extract_params(feature)["agg_type"]
        if agg_type == "count":
            return DataType.INT64
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Extract params from each feature, delegate to _compute_frame."""
        table = data

        for feature in features.features:
            feature_name = feature.name
            params = cls._extract_params(feature)

            mask_spec = parse_mask_spec(feature.options.get(MASK_KEY))

            table = cls._compute_frame(
                table,
                feature_name,
                params["source_col"],
                params["partition_by"],
                params["order_by"],
                params["agg_type"],
                params["frame_type"],
                params.get("frame_size"),
                params.get("frame_unit"),
                mask_spec,
            )

        return table

    @classmethod
    def _compute_frame(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
        agg_type: str,
        frame_type: str,
        frame_size: int | None = None,
        frame_unit: str | None = None,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> Any:
        """Subclasses must implement the actual frame computation."""
        raise NotImplementedError
