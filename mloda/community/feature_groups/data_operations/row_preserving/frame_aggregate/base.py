"""Base class for frame aggregate feature groups."""

from __future__ import annotations

import re
from typing import Any

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import DefaultOptionKeys, FeatureGroup


# Patterns for string-based feature names:
#   {col}__sum_rolling_3         -> rolling N rows
#   {col}__avg_5_day_window      -> time-interval window
#   {col}__cumsum                -> cumulative sum
#   {col}__expanding_avg         -> expanding window
_ROLLING_PATTERN = re.compile(r"^(.+)__(\w+)_rolling_(\d+)$")
_TIME_WINDOW_PATTERN = re.compile(r"^(.+)__(\w+)_(\d+)_(\w+)_window$")
_CUMULATIVE_PATTERN = re.compile(r"^(.+)__cum(\w+)$")
_EXPANDING_PATTERN = re.compile(r"^(.+)__expanding_(\w+)$")

_AGGREGATION_TYPES = {"sum", "avg", "count", "min", "max", "std", "var", "median"}
_CUMULATIVE_OPS = _AGGREGATION_TYPES
_TIME_UNITS = {"second", "minute", "hour", "day", "week", "month", "year"}


class FrameAggregateFeatureGroup(FeatureChainParserMixin, FeatureGroup):
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

    # Required by FeatureChainParserMixin but unused: match_feature_group_criteria
    # is fully overridden to handle four distinct patterns (rolling, time, cumulative,
    # expanding). This value is a placeholder to satisfy the mixin contract.
    PREFIX_PATTERN = r".*__([\w]+)_rolling_\d+$"

    SUPPORTED_FRAME_TYPES: set[str] = {"rolling", "time", "cumulative", "expanding"}

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    AGGREGATION_TYPE = "aggregation_type"
    FRAME_TYPE = "frame_type"
    FRAME_SIZE = "frame_size"
    FRAME_UNIT = "frame_unit"
    PARTITION_BY = "partition_by"
    ORDER_BY = "order_by"

    PROPERTY_MAPPING = {
        AGGREGATION_TYPE: {
            **{k: k for k in _AGGREGATION_TYPES},
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.default: None,
        },
        FRAME_TYPE: {
            "rolling": "Fixed-size row-count window",
            "time": "Time-interval window",
            "cumulative": "Running aggregate from start",
            "expanding": "Same as cumulative",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.default: None,
        },
        FRAME_SIZE: {
            "explanation": "Window size (rows for rolling, integer for time)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.default: None,
        },
        FRAME_UNIT: {
            "explanation": "Time unit for time windows",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.default: None,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature for frame aggregation",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
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
        m = _ROLLING_PATTERN.match(feature_name)
        if m:
            return {
                "source_col": m.group(1),
                "agg_type": m.group(2),
                "frame_type": "rolling",
                "frame_size": int(m.group(3)),
                "frame_unit": None,
            }

        m = _TIME_WINDOW_PATTERN.match(feature_name)
        if m:
            return {
                "source_col": m.group(1),
                "agg_type": m.group(2),
                "frame_type": "time",
                "frame_size": int(m.group(3)),
                "frame_unit": m.group(4),
            }

        m = _CUMULATIVE_PATTERN.match(feature_name)
        if m:
            return {
                "source_col": m.group(1),
                "agg_type": m.group(2),
                "frame_type": "cumulative",
                "frame_size": None,
                "frame_unit": None,
            }

        m = _EXPANDING_PATTERN.match(feature_name)
        if m:
            return {
                "source_col": m.group(1),
                "agg_type": m.group(2),
                "frame_type": "expanding",
                "frame_size": None,
                "frame_unit": None,
            }

        return None

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Any,
        options: Any,
        _data_access_collection: Any = None,
    ) -> bool:
        """Match using four string patterns or config-based matching.

        Validates:
        - Feature name matches one of the four patterns, OR config has required keys
        - partition_by is a list of strings
        - order_by is a string
        - aggregation_type is supported
        - For cumulative/expanding: agg in _CUMULATIVE_OPS (same as _AGGREGATION_TYPES)
        - For time: frame_unit in _TIME_UNITS
        """
        name = str(feature_name)

        parsed = cls._parse_frame_feature(name)

        if parsed is not None:
            if parsed["agg_type"] not in _AGGREGATION_TYPES:
                return False
            if parsed["frame_type"] in ("cumulative", "expanding") and parsed["agg_type"] not in _CUMULATIVE_OPS:
                return False
            if parsed["frame_type"] == "time" and parsed["frame_unit"] not in _TIME_UNITS:
                return False
            if parsed["frame_type"] not in cls.SUPPORTED_FRAME_TYPES:
                return False
        else:
            agg_type = options.get(cls.AGGREGATION_TYPE)
            frame_type = options.get(cls.FRAME_TYPE)
            if agg_type is None or frame_type is None:
                return False
            if str(agg_type) not in _AGGREGATION_TYPES:
                return False
            frame_type_str = str(frame_type)
            if frame_type_str not in cls.SUPPORTED_FRAME_TYPES:
                return False
            if frame_type_str in ("cumulative", "expanding") and str(agg_type) not in _CUMULATIVE_OPS:
                return False
            if frame_type_str == "time":
                frame_unit = options.get(cls.FRAME_UNIT)
                if frame_unit is None or str(frame_unit) not in _TIME_UNITS:
                    return False

        partition_by = options.get(cls.PARTITION_BY)
        if not isinstance(partition_by, (list, tuple)):
            return False
        if not all(isinstance(item, str) for item in partition_by):
            return False

        order_by = options.get(cls.ORDER_BY)
        if not isinstance(order_by, str):
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
                "order_by": feature.options.get(cls.ORDER_BY),
            }

        source_features = cls._extract_source_features(feature)
        return {
            "source_col": source_features[0],
            "agg_type": str(feature.options.get(cls.AGGREGATION_TYPE)),
            "frame_type": str(feature.options.get(cls.FRAME_TYPE)),
            "frame_size": feature.options.get(cls.FRAME_SIZE),
            "frame_unit": feature.options.get(cls.FRAME_UNIT),
            "partition_by": feature.options.get(cls.PARTITION_BY),
            "order_by": feature.options.get(cls.ORDER_BY),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Extract params from each feature, delegate to _compute_frame."""
        table = data

        for feature in features.features:
            feature_name = feature.name
            params = cls._extract_params(feature)

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
    ) -> Any:
        """Subclasses must implement the actual frame computation."""
        raise NotImplementedError
