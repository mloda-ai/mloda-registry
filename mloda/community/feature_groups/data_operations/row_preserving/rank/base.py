"""Base class for rank feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import DefaultOptionKeys, FeatureChainParser, FeatureGroup, FeatureSet, property_spec
from mloda.user import DataType, Feature, Options

from mloda.community.feature_groups.data_operations.base import (
    RejectionReasonMixin,
    column_ref_value,
    is_column_ref,
    is_in_features_value,
    is_op_token,
    is_parametric_suffix,
    op_token_value,
    option_value,
)
from mloda.community.feature_groups.data_operations.capability_hook import SubtypeCapabilityHook

_RANK_TYPES = {
    "row_number": "Sequential position, no ties",
    "rank": "Standard rank with gaps for ties",
    "dense_rank": "Rank without gaps for ties",
    "percent_rank": "Relative rank as fraction (0.0 to 1.0)",
}

#: Parametric rank families; each accepts a positive integer suffix (e.g. ``ntile_4``).
_PARAMETRIC_RANK_FAMILIES: tuple[str, ...] = ("ntile", "top", "bottom")


def _is_supported_rank_type(value: object) -> bool:
    """Fixed rank types plus parametric ntile_N / top_N / bottom_N with N >= 1."""
    if not isinstance(value, str):
        return False
    if value in _RANK_TYPES:
        return True
    for family in _PARAMETRIC_RANK_FAMILIES:
        prefix = f"{family}_"
        if value.startswith(prefix):
            if is_parametric_suffix(value[len(prefix) :]):
                return True
    return False


class RankFeatureGroup(SubtypeCapabilityHook, RejectionReasonMixin, FeatureGroup):
    """Base class for rank operations that preserve row count.

    Rank operations assign a rank or position to each row within a
    partition, ordered by a specified column. The output always has
    the same number of rows as the input.

    ## Supported Rank Types

    **Numeric rank types:**

    - ``row_number``: Sequential position (1, 2, 3, ...), no ties
    - ``rank``: Standard rank with gaps for ties (1, 2, 2, 4, ...)
    - ``dense_rank``: Rank without gaps (1, 2, 2, 3, ...)
    - ``percent_rank``: Relative rank as fraction from 0.0 to 1.0
    - ``ntile_N``: Divide rows into N roughly equal buckets (1..N)

    **Boolean mask types:**

    - ``top_N``: True if the row is in the top N values (ordered DESC, nulls last)
    - ``bottom_N``: True if the row is in the bottom N values (ordered ASC, nulls last)

    N must be a positive integer (>= 1). When N exceeds the partition size,
    all rows in that partition are True. Null values in the order column
    rank last in both directions and receive False when N < partition size.

    ## Feature Creation Methods

    ### 1. String-Based Creation

    Features follow the naming pattern: ``{source_column}__{rank_type}_ranked``

    Examples::

        features = [
            Feature("sales__row_number_ranked", options=Options(context={
                "partition_by": ["region"], "order_by": "sales",
            })),
            Feature("score__dense_rank_ranked", options=Options(context={
                "partition_by": ["team"], "order_by": "score",
            })),
            Feature("value__ntile_4_ranked", options=Options(context={
                "partition_by": ["region"], "order_by": "value",
            })),
            Feature("price__top_5_ranked", options=Options(context={
                "partition_by": ["category"], "order_by": "price",
            })),
            Feature("score__bottom_3_ranked", options=Options(context={
                "partition_by": ["team"], "order_by": "score",
            })),
        ]

    ### 2. Configuration-Based Creation

    Uses Options with proper context parameter separation::

        feature = Feature(
            name="my_rank",
            options=Options(
                context={
                    "rank_type": "row_number",
                    "in_features": "sales",
                    "partition_by": ["region"],
                    "order_by": "sales",
                }
            ),
        )

    ## Parameter Classification

    ### Context Parameters
    - ``rank_type``: The type of ranking to perform
    - ``in_features``: The source feature (used for ordering)
    - ``partition_by``: List of columns to partition by
    - ``order_by``: Column to order by within each partition
    """

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    RANK_TYPE = "rank_type"
    PARTITION_BY = "partition_by"
    ORDER_BY = "order_by"

    # Named after RANK_TYPE so bind_name_captures binds it, including the parametric families
    # (ntile_N / top_N / bottom_N) the old allowed_values-based fallback missed.
    PREFIX_PATTERN = rf".*__(?P<{RANK_TYPE}>[\w]+)_ranked$"

    # Aliases of the module tables the validator reads: overriding them in a subclass has no
    # effect, per-backend narrowing belongs in SubtypeCapabilityHook.
    RANK_TYPES = _RANK_TYPES

    PARAMETRIC_RANK_FAMILIES: tuple[str, ...] = _PARAMETRIC_RANK_FAMILIES

    PROPERTY_MAPPING = {
        RANK_TYPE: property_spec(
            "Rank type applied within each partition",
            strict=True,
            allowed_values=RANK_TYPES,
            element_validator=_is_supported_rank_type,
            match_guard=is_op_token,
        ),
        DefaultOptionKeys.in_features: property_spec(
            "Source feature for rank ordering",
            strict=False,
            match_guard=is_in_features_value,
        ),
        PARTITION_BY: property_spec(
            "List of columns to partition by",
            strict=False,
        ),
        ORDER_BY: property_spec(
            "Column to order by within each partition",
            strict=False,
            match_guard=is_column_ref,
        ),
    }

    @classmethod
    def _supports_rank_type(cls, rank_type: str) -> bool:
        """Check if the given rank type is supported, including ntile_N, top_N, and bottom_N."""
        # Per-backend narrowing lives in SubtypeCapabilityHook, not here.
        return _is_supported_rank_type(rank_type)

    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, source_feature: str) -> bool:
        """Validate that the rank type is supported (including ntile_N, top_N, bottom_N)."""
        return cls._supports_rank_type(operation_config)

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Any,
        options: Any,
        _data_access_collection: Any = None,
    ) -> bool:
        """Extend mixin matching with partition_by, order_by, and in_features validation."""
        if not super().match_feature_group_criteria(feature_name, options, _data_access_collection):
            return False

        partition_by = options.get(cls.PARTITION_BY)
        if partition_by is None:
            return False
        if not isinstance(partition_by, (list, tuple)):
            return False
        if not partition_by:
            return False
        if not all(isinstance(item, str) for item in partition_by):
            return False

        if not is_column_ref(options.get(cls.ORDER_BY)):
            return False

        in_features_raw = options.get(DefaultOptionKeys.in_features)
        if in_features_raw is not None:
            in_features = options.get_in_features()
            if len(in_features) > cls.MAX_IN_FEATURES:
                return False

        return True

    @classmethod
    def get_rank_type(cls, feature_name: str) -> str:
        """Extract the rank type from a feature name string."""
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        raise ValueError(f"Could not extract rank type from feature name: {feature_name}")

    @classmethod
    def _extract_rank_type(cls, feature: Feature) -> str:
        """Extract rank type from feature (string-based or config-based)."""
        feature_name = feature.name
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            return operation_config
        rank_type = feature.options.get(cls.RANK_TYPE)
        if rank_type is None:
            raise ValueError(f"Could not extract rank type for {feature_name}")
        return op_token_value(rank_type)

    @classmethod
    def _resolve_rank_type(cls, feature_name: str, options: Options) -> str | None:
        """Resolve the rank type from the feature name or options; None if unresolvable."""
        try:
            operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, cls._get_prefix_patterns())
        except ValueError:
            return None
        if operation_config is not None:
            return operation_config
        rank_type = options.get(cls.RANK_TYPE)
        return None if rank_type is None else op_token_value(rank_type)

    @classmethod
    def _capability_subtype(cls, feature_name: str, options: Options) -> str | None:
        return cls._resolve_rank_type(feature_name, options)

    @classmethod
    def _capability_restrictable(cls, subtype: str) -> bool:
        # Parametric families (ntile_N, top_N, bottom_N) stay open; only named types are checked.
        return subtype in cls.RANK_TYPES

    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> DataType | None:
        """Declare deterministic rank output types.

        row_number / rank / dense_rank / ntile_N are integer ranks (INT64);
        percent_rank is a fractional rank (DOUBLE). top_N / bottom_N (and any
        unparseable input) stay open and return None.
        """
        rank_type = cls._extract_rank_type(feature)
        if rank_type in {"row_number", "rank", "dense_rank"}:
            return DataType.INT64
        if rank_type == "percent_rank":
            return DataType.DOUBLE
        if rank_type.startswith("ntile_") and is_parametric_suffix(rank_type[len("ntile_") :]):
            return DataType.INT64
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Shared loop: extract params from each feature, delegate to _compute_rank.

        Supports both string-based features (e.g. "value_int__row_number_ranked") and
        configuration-based features (via Options with rank_type, in_features,
        partition_by, order_by).
        """
        table = data

        for feature in features.features:
            feature_name = feature.name

            rank_type = cls._extract_rank_type(feature)
            partition_by = feature.options.get(cls.PARTITION_BY)
            if not isinstance(partition_by, (list, tuple)) or not partition_by:
                raise ValueError(
                    f"rank requires a non-empty partition_by, got {partition_by!r} for feature {feature_name!r}."
                )
            partition_by = list(partition_by)
            # Any: matching requires order_by, but a direct call still passes an absent one through.
            order_by: Any = option_value(feature.options, cls.ORDER_BY, column_ref_value)

            table = cls._compute_rank(table, feature_name, partition_by, order_by, rank_type)

        return table

    @classmethod
    def _compute_rank(
        cls,
        data: Any,
        feature_name: str,
        partition_by: list[str],
        order_by: str,
        rank_type: str,
    ) -> Any:
        """Subclasses must implement the actual rank computation."""
        raise NotImplementedError
