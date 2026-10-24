"""Base class for frame aggregate feature groups."""

from __future__ import annotations

import functools
import re
from collections.abc import Callable
from typing import Any

from mloda.provider import (
    DefaultOptionKeys,
    FeatureChainParserMixin,
    FeatureGroup,
    FeatureSet,
    OperationFeatureGroup,
    SubtypeDeclaration,
    escalate_match_abort,
    property_spec,
    record_match_rejection,
)
from mloda.user import DataType, Feature, Options

from mloda.community.feature_groups.data_operations.base import (
    COLUMN_REF_EXPECTED,
    IN_FEATURES_EXPECTED,
    OP_TOKEN_EXPECTED,
    POSITIVE_INT_EXPECTED,
    KeyColumnInputsMixin,
    always_required,
    assert_key_columns_present,
    assert_source_columns_present,
    column_ref_value,
    is_column_ref,
    is_in_features_value,
    is_op_token,
    is_own_option,
    is_positive_int,
    op_token_value,
    option_value,
    positive_int_value,
    validate_hand_parsed_bindings,
)
from mloda.community.feature_groups.data_operations.base import (
    FRAME_SIZE as _FRAME_SIZE_KEY,
)
from mloda.community.feature_groups.data_operations.base import (
    FRAME_TYPE as _FRAME_TYPE_KEY,
)
from mloda.community.feature_groups.data_operations.base import (
    FRAME_UNIT as _FRAME_UNIT_KEY,
)
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
# Options each frame type uses; any other own frame_size/frame_unit is rejected.
_FRAME_OPTIONS_USED: dict[str, frozenset[str]] = {
    "rolling": frozenset({_FRAME_SIZE_KEY}),
    "time": frozenset({_FRAME_SIZE_KEY, _FRAME_UNIT_KEY}),
    "cumulative": frozenset(),
    "expanding": frozenset(),
}
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

    m = _ROLLING_PATTERN.fullmatch(feature_name)
    if m:
        return {
            "source_col": source_col,
            "agg_type": m.group(1),
            "frame_type": "rolling",
            "frame_size": int(m.group(2)),
            "frame_unit": None,
        }

    m = _TIME_WINDOW_PATTERN.fullmatch(feature_name)
    if m:
        return {
            "source_col": source_col,
            "agg_type": m.group(1),
            "frame_type": "time",
            "frame_size": int(m.group(2)),
            "frame_unit": m.group(3),
        }

    m = _CUMULATIVE_PATTERN.fullmatch(feature_name)
    if m:
        return {
            "source_col": source_col,
            "agg_type": m.group(1),
            "frame_type": "cumulative",
            "frame_size": None,
            "frame_unit": None,
        }

    m = _EXPANDING_PATTERN.fullmatch(feature_name)
    if m:
        return {
            "source_col": source_col,
            "agg_type": m.group(1),
            "frame_type": "expanding",
            "frame_size": None,
            "frame_unit": None,
        }

    return None


def _resolve_frame_agg_type(feature_name: str, options: Options) -> str | None:
    """The aggregation type from any of the four name shapes, else the ``aggregation_type`` option."""
    parsed = _parse_frame_feature_cached(feature_name)
    if parsed is not None:
        return str(parsed["agg_type"])
    agg_type = options.get("aggregation_type")
    return None if agg_type is None else op_token_value(agg_type)


class FrameAggregateFeatureGroup(OperationFeatureGroup, KeyColumnInputsMixin, FeatureChainParserMixin, FeatureGroup):
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

    SUBTYPES = SubtypeDeclaration(universe=_AGGREGATION_TYPES, resolver=_resolve_frame_agg_type)

    KEY_COLUMN_OPTIONS = ("partition_by", "order_by", "mask")

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

    AGGREGATION_TYPE = "aggregation_type"
    # Aliases of the shared keys, so the literals stay defined once in data_operations/base.py.
    FRAME_TYPE = _FRAME_TYPE_KEY
    FRAME_SIZE = _FRAME_SIZE_KEY
    FRAME_UNIT = _FRAME_UNIT_KEY
    PARTITION_BY = "partition_by"
    ORDER_BY = "order_by"

    PROPERTY_MAPPING = {
        AGGREGATION_TYPE: property_spec(
            "Aggregation applied over the frame",
            strict=True,
            allowed_values={k: k for k in _AGGREGATION_TYPES},
            match_guard=is_op_token,
            expected=OP_TOKEN_EXPECTED,
        ),
        FRAME_TYPE: property_spec(
            "Frame semantics of the window",
            strict=True,
            allowed_values={
                "rolling": "Fixed-size row-count window",
                "time": "Time-interval window",
                "cumulative": "Running aggregate from start",
                "expanding": "Alias of cumulative",
            },
            match_guard=is_op_token,
            expected=OP_TOKEN_EXPECTED,
            # frame_type is inferred from which pattern shape matched, not captured text, so the binding is built by hand.
            deferred_binding=True,
        ),
        FRAME_SIZE: property_spec(
            "Window size (rows for rolling, integer for time)",
            match_guard=is_positive_int,
            expected=POSITIVE_INT_EXPECTED,
            # Optional by declaration: conditional requiredness (rolling/time only) is hand-enforced
            # in match_feature_group_criteria's config-path branch, not via required_when. A
            # required_when predicate here would also fire on the name path through core's
            # class-definition-time required_when guard, which evaluates it against Options alone
            # (with no notion of "this feature name already supplies the value"), rejecting a
            # name-parsed feature that merely carries a stray/matching frame_type option.
            default=None,
        ),
        FRAME_UNIT: property_spec(
            "Time unit for time windows",
            strict=True,
            allowed_values={unit: f"{unit} interval" for unit in sorted(_TIME_UNITS)},
            match_guard=is_op_token,
            expected=OP_TOKEN_EXPECTED,
            # Same rationale as FRAME_SIZE above: conditional requiredness is hand-enforced below.
            default=None,
        ),
        DefaultOptionKeys.in_features: property_spec(
            "Source feature for frame aggregation",
            match_guard=is_in_features_value,
            expected=IN_FEATURES_EXPECTED,
        ),
        PARTITION_BY: property_spec(
            "List of columns to partition by",
        ),
        ORDER_BY: property_spec(
            "Column to order by within each partition",
            match_guard=is_column_ref,
            expected=COLUMN_REF_EXPECTED,
            required_when=always_required,
        ),
        MASK_KEY: property_spec(
            "Conditional mask: (column, operator, value) tuple or list of tuples",
            default=None,
        ),
    }

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

        parsed = cls._parse_frame_feature(str(feature_name))
        if parsed is not None:
            cls._validate_frame_name_agreement(feature_name, options, parsed)

        # Config path only: SUPPORTED_* narrows per subclass, so it cannot be declared once on the base.
        # frame_size/frame_unit requiredness is hand-enforced here (not via PROPERTY_MAPPING
        # required_when) so it can never mis-fire on the name path; see the PROPERTY_MAPPING comment.
        if parsed is None:
            frame_type = _option_token(options, cls.FRAME_TYPE)
            if frame_type not in cls.SUPPORTED_FRAME_TYPES:
                return False
            if not is_own_option(options, cls.FRAME_TYPE):
                record_match_rejection(
                    cls.get_class_name(),
                    f"option '{cls.FRAME_TYPE}' is inherited, not set on the feature; "
                    "set it on the feature to use frame aggregation",
                )
                return False
            cls._reject_unused_frame_options(feature_name, options, str(frame_type))
            if _needs_frame_size(options) and options.get(cls.FRAME_SIZE) is None:
                record_match_rejection(
                    cls.get_class_name(),
                    f"required option '{cls.FRAME_SIZE}' is absent for frame_type '{frame_type}'",
                )
                return False
            if _needs_frame_unit(options) and options.get(cls.FRAME_UNIT) is None:
                record_match_rejection(
                    cls.get_class_name(),
                    f"required option '{cls.FRAME_UNIT}' is absent for frame_type '{frame_type}'",
                )
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
    def _reject_unused_frame_options(cls, feature_name: Any, options: Any, frame_type: str) -> None:
        """Raise on an own frame_size/frame_unit the frame type does not use."""
        used = _FRAME_OPTIONS_USED.get(frame_type)
        if used is None:
            return
        for key in (cls.FRAME_SIZE, cls.FRAME_UNIT):
            if key not in used and is_own_option(options, key):
                msg = f"Feature '{feature_name}': option '{key}' is not used by a '{frame_type}' frame; remove the option."
                # Marked: user misconfiguration; the option would be silently ignored otherwise.
                raise escalate_match_abort(ValueError(msg))

    @classmethod
    def _validate_frame_name_agreement(cls, feature_name: Any, options: Any, parsed: dict[str, Any]) -> None:
        """Reject an own or forwarded frame_type/frame_size/frame_unit that contradicts the name."""
        bindings = {
            key: str(parsed[field])
            for key, field in (
                (cls.FRAME_TYPE, "frame_type"),
                (cls.FRAME_SIZE, "frame_size"),
                (cls.FRAME_UNIT, "frame_unit"),
            )
            if parsed[field] is not None
        }
        name_type = parsed["frame_type"]
        if name_type in ("cumulative", "expanding"):
            declared = _option_token(options, cls.FRAME_TYPE)
            # Aliases: either agrees with either name.
            if declared in ("cumulative", "expanding"):
                bindings[cls.FRAME_TYPE] = declared
        validate_hand_parsed_bindings(cls, feature_name, bindings, options)
        cls._reject_unused_frame_options(feature_name, options, str(name_type))

    @classmethod
    def frame_accepts(
        cls,
        feature_name: str,
        options: Options,
        *,
        time_units: set[str] | frozenset[str] = frozenset(_TIME_UNITS),
        agg_types_by_frame: dict[str, frozenset[str]] | None = None,
    ) -> bool:
        """For Implementations' ``accepts``: reject a time unit outside ``time_units`` and an aggregation
        outside ``agg_types_by_frame[frame_type]``; anything unresolved stays True."""
        parsed = cls._parse_frame_feature(feature_name)
        if parsed is not None:
            frame_type, frame_unit, agg_type = parsed["frame_type"], parsed["frame_unit"], parsed["agg_type"]
        else:
            raw_type = options.get(cls.FRAME_TYPE)
            frame_type = None if raw_type is None else op_token_value(raw_type)
            frame_unit = _option_token(options, cls.FRAME_UNIT)
            agg_type = _resolve_frame_agg_type(feature_name, options)
        if frame_type == "time" and frame_unit is not None and frame_unit not in time_units:
            return False
        if agg_types_by_frame is not None and frame_type in agg_types_by_frame and agg_type is not None:
            return bool(agg_type in agg_types_by_frame[frame_type])
        return True

    @classmethod
    def _extract_params(cls, feature: Feature) -> dict[str, Any]:
        """Extract all frame aggregate parameters from a feature."""
        feature_name = feature.name
        parsed = cls._parse_frame_feature(feature_name)

        # Compute-time backstop for direct calls; match time enforces presence declaratively.
        order_by = option_value(feature.options, cls.ORDER_BY, column_ref_value)
        if order_by is None:
            raise ValueError("frame_aggregate requires an 'order_by' column in Options context.")

        if parsed is not None:
            return {
                "source_col": parsed["source_col"],
                "agg_type": parsed["agg_type"],
                "frame_type": parsed["frame_type"],
                "frame_size": parsed["frame_size"],
                "frame_unit": parsed["frame_unit"],
                "partition_by": feature.options.get(cls.PARTITION_BY),
                "order_by": order_by,
            }

        return {
            "source_col": cls._extract_single_source_feature(feature),
            "agg_type": op_token_value(feature.options.get(cls.AGGREGATION_TYPE)),
            "frame_type": op_token_value(feature.options.get(cls.FRAME_TYPE)),
            "frame_size": option_value(feature.options, cls.FRAME_SIZE, positive_int_value),
            "frame_unit": _option_token(feature.options, cls.FRAME_UNIT),
            "partition_by": feature.options.get(cls.PARTITION_BY),
            "order_by": order_by,
        }

    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> DataType | None:
        """Declare INT64 for count (a counting op); other aggregates stay open."""
        agg_type = cls._extract_params(feature)["agg_type"]
        if agg_type == "count":
            return DataType.INT64
        return None

    @classmethod
    def aggregate(
        cls,
        data: Any,
        features: FeatureSet,
        *,
        compute_frame: Callable[..., Any],
    ) -> Any:
        """Shared loop: extract params from each feature, delegate to ``compute_frame``."""
        table = data

        for feature in features.features:
            feature_name = feature.name
            params = cls._extract_params(feature)
            assert_source_columns_present(data, [params["source_col"]])

            mask_spec = parse_mask_spec(feature.options.get(MASK_KEY))
            assert_key_columns_present(data, params["partition_by"], params["order_by"], mask_spec)

            table = compute_frame(
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
