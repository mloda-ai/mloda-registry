"""Base class for key-lookup feature groups.

Replaces each non-null source value with the value its key maps to in a mapping.
The operation is ROW-PRESERVING: the result has the same rows in the same order as
the input, with one new ``{col}__lookup`` column appended.

Pattern: ``{src}__lookup``

Options context:

- ``lookup_mapping``: non-empty mapping; keys all ``str`` or all ``int``, values non-null
  and all one family (str, int, float or bool). Required for the generic operation.
- ``on_unmatched``: ``raise`` (default), ``null`` or ``keep`` for a non-null source value
  that is not a key. ``keep`` needs key family == value family.
- ``in_features``: the single source column (when not derivable from the name).

Null source values stay null under every policy and are never counted as unmatched.

Plugins supply their own mapping by setting ``MAPPING_FROM_OPTION = False``, their own
``PREFIX_PATTERN`` and overriding ``resolve_mapping`` (and optionally
``resolve_unmatched_reasons``); such a subclass matches by name only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, ClassVar

from mloda.provider import (
    DefaultOptionKeys,
    FeatureChainParserMixin,
    FeatureGroup,
    FeatureSet,
    property_spec,
)
from mloda.user import DataType, Feature

from mloda.community.feature_groups.data_operations.base import (
    assert_source_columns_present,
    op_token_value,
    option_value,
)

ON_UNMATCHED_POLICIES: dict[str, str] = {
    "raise": "Raise LookupUnmatchedError naming the unmatched keys",
    "null": "Output null for an unmatched key",
    "keep": "Pass the unmatched source value through (key family must equal value family)",
}

SAMPLE_SIZE = 10

_VALUE_DATA_TYPES: dict[str, DataType] = {
    "str": DataType.STRING,
    "int": DataType.INT64,
    "float": DataType.DOUBLE,
    "bool": DataType.BOOLEAN,
}

MAPPING_EXPECTED = (
    "a non-empty mapping with all-str or all-int keys and non-null values of one type (str, int, float or bool)"
)


class LookupUnmatchedError(ValueError):
    """Non-null source values that are not keys of the mapping (policy ``raise``)."""

    def __init__(
        self,
        message: str,
        feature_name: str,
        unmatched_rows: int,
        distinct_keys: int,
        sample: tuple[Any, ...],
        reasons: Mapping[Any, str],
    ) -> None:
        super().__init__(message)
        self.feature_name = feature_name
        self.unmatched_rows = unmatched_rows
        self.distinct_keys = distinct_keys
        self.sample = sample
        self.reasons = reasons


def _value_family(value: object) -> str | None:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return None


def _key_family(key: object) -> str | None:
    family = _value_family(key)
    return family if family in ("str", "int") else None


def mapping_problem(mapping: object) -> str | None:
    """Why ``mapping`` is not a usable lookup mapping, or None when it is valid."""
    if not isinstance(mapping, Mapping):
        return "must be a mapping"
    if not mapping:
        return "must not be empty"
    key_families = {_key_family(key) for key in mapping}
    if len(key_families) != 1 or None in key_families:
        return "keys must be all str or all int (no null, bool or float)"
    value_families = {_value_family(value) for value in mapping.values()}
    if len(value_families) != 1 or None in value_families:
        return "values must be non-null and all one type (str, int, float or bool)"
    return None


def is_valid_mapping(value: object) -> bool:
    return mapping_problem(value) is None


def key_family(mapping: Mapping[Any, Any]) -> str:
    """``str`` or ``int`` for a valid mapping."""
    return str(_key_family(next(iter(mapping))))


def value_family(mapping: Mapping[Any, Any]) -> str:
    """``str``, ``int``, ``float`` or ``bool`` for a valid mapping."""
    return str(_value_family(next(iter(mapping.values()))))


def assert_key_family(feature_name: str, source_col: str, mapping: Mapping[Any, Any], source_family: str) -> None:
    """Raise ValueError when the mapping key family differs from the source column family."""
    expected = key_family(mapping)
    if source_family != expected:
        raise ValueError(
            f"{feature_name}: mapping keys are {expected} but source column '{source_col}' is {source_family}"
        )


def build_unmatched_error(
    feature_name: str,
    unmatched_rows: int,
    distinct_keys: Iterable[Any],
    reasons: Mapping[Any, str],
) -> LookupUnmatchedError:
    """Build the error from the unmatched row count and the distinct unmatched keys."""
    keys = sorted(distinct_keys)
    sample = tuple(keys[:SAMPLE_SIZE])
    sample_reasons = {key: reasons[key] for key in sample if key in reasons}
    shown = [f"{key!r} ({sample_reasons[key]})" if key in sample_reasons else repr(key) for key in sample]
    message = (
        f"{feature_name}: {unmatched_rows} rows, {len(keys)} distinct keys not in the mapping. "
        f"Sample: [{', '.join(shown)}]"
    )
    return LookupUnmatchedError(message, feature_name, unmatched_rows, len(keys), sample, sample_reasons)


class LookupFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Base class for key-lookup operations that preserve row count."""

    PREFIX_PATTERN = r".*__lookup$"
    RECOGNITION_ONLY_PATTERN = True

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    LOOKUP_MAPPING = "lookup_mapping"
    ON_UNMATCHED = "on_unmatched"

    # False: the subclass resolves its own mapping and matches by name only.
    MAPPING_FROM_OPTION: ClassVar[bool] = True

    PROPERTY_MAPPING = {
        DefaultOptionKeys.in_features: property_spec(
            "Single source column to look up",
        ),
        LOOKUP_MAPPING: property_spec(
            "Mapping from source keys to output values",
            default=None,
            match_guard=is_valid_mapping,
            expected=MAPPING_EXPECTED,
        ),
        ON_UNMATCHED: property_spec(
            "Policy for a non-null source value that is not a key",
            strict=True,
            allowed_values=ON_UNMATCHED_POLICIES,
            default="raise",
        ),
    }

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.MAPPING_FROM_OPTION and cls.PREFIX_PATTERN == LookupFeatureGroup.PREFIX_PATTERN:
            raise TypeError(f"{cls.__name__}: MAPPING_FROM_OPTION = False requires its own PREFIX_PATTERN")

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Any,
        options: Any,
        data_access_collection: Any = None,
    ) -> bool:
        if not super().match_feature_group_criteria(feature_name, options, data_access_collection):
            return False
        if cls.MAPPING_FROM_OPTION:
            return options.get(cls.LOOKUP_MAPPING) is not None
        return cls.resolve_feature_name(feature_name).owned

    @classmethod
    def resolve_mapping(cls, feature: Feature) -> Mapping[Any, Any]:
        """The lookup mapping for ``feature``; plugins override this."""
        mapping: Mapping[Any, Any] = feature.options.get(cls.LOOKUP_MAPPING)
        return mapping

    @classmethod
    def resolve_unmatched_reasons(cls, feature: Feature) -> Mapping[Any, str]:
        """Optional per-key reasons rendered in the unmatched error."""
        return {}

    @classmethod
    def _validated_mapping(cls, feature: Feature) -> Mapping[Any, Any]:
        mapping = cls.resolve_mapping(feature)
        problem = mapping_problem(mapping)
        if problem is not None:
            raise ValueError(f"{feature.name}: invalid lookup mapping, {problem}")
        return mapping

    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> DataType | None:
        """Value family of the mapping; None for an invalid mapping."""
        mapping = cls.resolve_mapping(feature)
        if mapping_problem(mapping) is not None:
            return None
        return _VALUE_DATA_TYPES[value_family(mapping)]

    @classmethod
    def _extract_policy(cls, feature: Feature) -> str:
        return option_value(feature.options, cls.ON_UNMATCHED, op_token_value) or "raise"

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        table = data

        for feature in features.features:
            feature_name = str(feature.name)

            source_col = cls._extract_single_source_feature(feature)
            assert_source_columns_present(data, [source_col])
            mapping = cls._validated_mapping(feature)
            policy = cls._extract_policy(feature)
            reasons = cls.resolve_unmatched_reasons(feature)
            if policy == "keep" and key_family(mapping) != value_family(mapping):
                raise ValueError(
                    f"{feature_name}: on_unmatched='keep' needs keys and values of the same type, "
                    f"got {key_family(mapping)} keys and {value_family(mapping)} values"
                )

            table = cls._compute_lookup(table, feature_name, source_col, mapping, policy, reasons)

        return table

    @classmethod
    def _compute_lookup(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        mapping: Mapping[Any, Any],
        policy: str,
        reasons: Mapping[Any, str],
    ) -> Any:
        raise NotImplementedError
