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

import math
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any, ClassVar

from mloda.provider import (
    DefaultOptionKeys,
    FeatureChainParserMixin,
    FeatureGroup,
    FeatureSet,
    OperationFeatureGroup,
    escalate_match_abort,
    property_spec,
)
from mloda.user import DataType, Feature

from mloda.community.feature_groups.data_operations.base import (
    OP_TOKEN_EXPECTED,
    assert_source_columns_present,
    is_op_token,
    op_token_value,
    option_value,
)

ON_UNMATCHED_POLICIES: dict[str, str] = {
    "raise": "Raise LookupUnmatchedError naming the unmatched keys",
    "null": "Output null for an unmatched key",
    "keep": "Pass the unmatched source value through (key family must equal value family)",
}

SAMPLE_SIZE = 10
DEFAULT_POLICY = "raise"

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

    def __reduce__(self) -> tuple[Any, ...]:
        args = (self.args[0], self.feature_name, self.unmatched_rows, self.distinct_keys, self.sample, self.reasons)
        return (type(self), args, self.__dict__)


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


_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def _outside_int64(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and not _INT64_MIN <= value <= _INT64_MAX


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
    if any(isinstance(value, float) and math.isnan(value) for value in mapping.values()):
        return "values must not be NaN"
    if any(_outside_int64(item) for item in (*mapping, *mapping.values())):
        return "int keys and values must fit in int64"
    return None


def is_valid_mapping(value: object) -> bool:
    return mapping_problem(value) is None


def key_family(mapping: Mapping[Any, Any]) -> str:
    """``str`` or ``int`` for a valid mapping."""
    family = _key_family(next(iter(mapping), None))
    if family is None:
        raise ValueError("invalid lookup mapping, keys must be all str or all int")
    return family


def value_family(mapping: Mapping[Any, Any]) -> str:
    """``str``, ``int``, ``float`` or ``bool`` for a valid mapping."""
    family = _value_family(next(iter(mapping.values()), None))
    if family is None:
        raise ValueError("invalid lookup mapping, values must be non-null str, int, float or bool")
    return family


def assert_key_family(
    feature_name: str,
    source_col: str,
    mapping: Mapping[Any, Any],
    source_family: str,
    source_dtype: str | None = None,
) -> None:
    """Raise ValueError when the mapping key family differs from the source column family."""
    expected = key_family(mapping)
    if source_family != expected:
        shown = f"has type {source_dtype}" if source_dtype is not None else f"is {source_family}"
        raise ValueError(f"{feature_name}: mapping keys are {expected} but source column '{source_col}' {shown}")


def keep_family_message(feature_name: str, mapping: Mapping[Any, Any]) -> str:
    return (
        f"{feature_name}: on_unmatched='keep' needs keys and values of the same type, "
        f"got {key_family(mapping)} keys and {value_family(mapping)} values"
    )


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


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
        f"{feature_name}: {_plural(unmatched_rows, 'row')}, {_plural(len(keys), 'distinct key')} not in the mapping. "
        f"Sample: [{', '.join(shown)}]"
    )
    return LookupUnmatchedError(message, feature_name, unmatched_rows, len(keys), sample, sample_reasons)


class LookupFeatureGroup(OperationFeatureGroup, FeatureChainParserMixin, FeatureGroup):
    """Base class for key-lookup operations that preserve row count.

    ## Reference Contract

    PyArrowLookup is the reference. Null keys stay null under every policy and never count as unmatched.
    The output type follows the mapping's value family.
    """

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
            default=DEFAULT_POLICY,
            match_guard=is_op_token,
            expected=OP_TOKEN_EXPECTED,
        ),
    }

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.MAPPING_FROM_OPTION and cls.PREFIX_PATTERN == LookupFeatureGroup.PREFIX_PATTERN:
            raise TypeError(f"{cls.__name__}: MAPPING_FROM_OPTION = False requires its own PREFIX_PATTERN")
        if cls.MAPPING_FROM_OPTION:
            if LookupFeatureGroup.LOOKUP_MAPPING not in cls.PROPERTY_MAPPING:
                # A subclass of a hook that flips the flag back needs the option spec again.
                cls.PROPERTY_MAPPING = {
                    **cls.PROPERTY_MAPPING,
                    LookupFeatureGroup.LOOKUP_MAPPING: LookupFeatureGroup.PROPERTY_MAPPING[
                        LookupFeatureGroup.LOOKUP_MAPPING
                    ],
                }
        else:
            # A hook supplies its own mapping, so the option spec (and its guard) must not apply.
            cls.PROPERTY_MAPPING = {
                k: v for k, v in cls.PROPERTY_MAPPING.items() if k != LookupFeatureGroup.LOOKUP_MAPPING
            }

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Any,
        options: Any,
        data_access_collection: Any = None,
    ) -> bool:
        if not cls.MAPPING_FROM_OPTION:
            cls._reject_own_mapping_option(feature_name, options)
        if not super().match_feature_group_criteria(feature_name, options, data_access_collection):
            return False
        if not cls.MAPPING_FROM_OPTION:
            return cls.resolve_feature_name(feature_name).owned
        mapping = options.get(cls.LOOKUP_MAPPING)
        if mapping is None:
            return False
        if cls._extract_policy_from_options(options) == "keep" and key_family(mapping) != value_family(mapping):
            # Marked: user misconfiguration; containing it would let a rival group win.
            raise escalate_match_abort(ValueError(keep_family_message(str(feature_name), mapping)))
        return True

    @classmethod
    def _reject_own_mapping_option(cls, feature_name: Any, options: Any) -> None:
        if not cls.resolve_feature_name(feature_name).owned:
            return
        if options.get(cls.LOOKUP_MAPPING) is None:
            return
        if cls.LOOKUP_MAPPING in (options.inherited_group_keys or ()):
            return
        if cls.LOOKUP_MAPPING in (options.inherited_context_keys or ()):
            return
        # A generic lookup source owns the group mapping (e.g. ``region__lookup__regions``).
        if any(
            re.match(LookupFeatureGroup.PREFIX_PATTERN, source)
            for source in cls.resolve_feature_name(feature_name).sources
        ):
            return
        message = (
            f"Feature '{feature_name}': option '{cls.LOOKUP_MAPPING}' is set but {cls.__name__} supplies its own "
            f"mapping. Remove the '{cls.LOOKUP_MAPPING}' option."
        )
        # Marked: user misconfiguration; the option would be silently ignored otherwise.
        raise escalate_match_abort(ValueError(message))

    @classmethod
    def _extract_policy_from_options(cls, options: Any) -> str:
        return option_value(options, cls.ON_UNMATCHED, op_token_value) or DEFAULT_POLICY

    @classmethod
    def resolve_mapping(cls, feature: Feature) -> Mapping[Any, Any] | None:
        """The lookup mapping for ``feature``; plugins override this."""
        mapping: Mapping[Any, Any] | None = feature.options.get(cls.LOOKUP_MAPPING)
        return mapping

    @classmethod
    def resolve_unmatched_reasons(cls, feature: Feature) -> Mapping[Any, str]:
        """Optional per-key reasons rendered in the unmatched error."""
        return {}

    @classmethod
    def _validated_mapping(cls, feature: Feature) -> Mapping[Any, Any]:
        mapping = cls.resolve_mapping(feature)
        problem = mapping_problem(mapping)
        if problem is not None or mapping is None:
            raise ValueError(f"{feature.name}: invalid lookup mapping, {problem}")
        return mapping

    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> DataType | None:
        """Value family of the mapping; None for an invalid mapping."""
        mapping = cls.resolve_mapping(feature)
        if mapping is None or mapping_problem(mapping) is not None:
            return None
        return _VALUE_DATA_TYPES[value_family(mapping)]

    @classmethod
    def _extract_policy(cls, feature: Feature) -> str:
        return cls._extract_policy_from_options(feature.options)

    @classmethod
    def lookup(
        cls,
        data: Any,
        features: FeatureSet,
        *,
        compute_lookup: Callable[[Any, str, str, Mapping[Any, Any], str, Mapping[Any, Any]], Any],
    ) -> Any:
        """Shared loop: validate each feature, delegate to ``compute_lookup``."""
        table = data

        for feature in features.features:
            feature_name = str(feature.name)

            source_col = cls._extract_single_source_feature(feature)
            assert_source_columns_present(data, [source_col])
            mapping = cls._validated_mapping(feature)
            policy = cls._extract_policy(feature)
            reasons = cls.resolve_unmatched_reasons(feature)
            if policy == "keep" and key_family(mapping) != value_family(mapping):
                raise ValueError(keep_family_message(feature_name, mapping))

            table = compute_lookup(table, feature_name, source_col, mapping, policy, reasons)

        return table
