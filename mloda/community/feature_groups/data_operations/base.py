"""Shared constants and utilities for all data operation feature groups."""

from __future__ import annotations

import logging
import re
import reprlib
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    option_key_is_present,
)  # no public equivalent yet
from mloda.core.abstract_plugins.components.utils import (
    contained_raise_log_level,
    contained_raise_reason,
)  # no public equivalent yet
from mloda.provider import FeatureChainParser, FeatureChainParserMixin, PropertySpec
from mloda.user import FeatureName, Options

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Shared config key constants
# ---------------------------------------------------------------------------
# These are the keys used in Options to configure data operation feature groups.
# Core's DefaultOptionKeys defines standard keys like "order_by".
# Use DefaultOptionKeys directly for keys that exist in core.
# The constants below are data-operations-specific keys not in DefaultOptionKeys.

PARTITION_BY = "partition_by"
"""Config key for partitioning columns (list of column names).

Used by: window_aggregation, aggregation, rank, offset, frame_aggregate.
"""

FRAME_TYPE = "frame_type"
"""Config key for frame type in frame_aggregate operations.

Valid values: "rows", "time", "expanding", "cumulative".
"""

FRAME_SIZE = "frame_size"
"""Config key for frame size (positive integer).

Used by frame_aggregate with frame_type "rows" or "time".
"""

FRAME_UNIT = "frame_unit"
"""Config key for time unit in time-interval frames.

Used by frame_aggregate with frame_type "time".
Valid values: second, minute, hour, day, week, month, year.
"""


# ---------------------------------------------------------------------------
# Null handling policy constants
# ---------------------------------------------------------------------------
# These document the data operations null handling contract.
# Implementations must match these defaults. PyArrow behavior is the reference.


class NullPolicy(str, Enum):
    """Null handling behavior constants for data operations.

    Each value describes a null handling rule. Implementations must match
    PyArrow's behavior as the reference. Where a framework diverges from
    these defaults, add explicit convergence code (e.g. pandas groupby
    needs ``dropna=False``; SQLite rank needs an explicit null-last clause).

    These constants are documentation and configuration anchors, not
    runtime enforcement. Each package's ``calculate_feature`` is responsible
    for honoring the policy.
    """

    PROPAGATE = "propagate"
    """Element-wise operations return null for null input (null in, null out).

    Applies to: datetime, string, binning.
    """

    SKIP = "skip"
    """Aggregations skip null values (e.g. SUM ignores nulls).

    Applies to: window_aggregation, aggregation, frame_aggregate.
    """

    NULL_IS_GROUP = "null_is_group"
    """Null is a valid group key in partitioned operations.

    Applies to: window_aggregation, aggregation, rank, offset, frame_aggregate.
    Pandas divergence: pass ``dropna=False`` to ``groupby()``.
    """

    NULLS_LAST = "nulls_last"
    """Nulls rank last in ordered operations.

    Applies to: rank.
    SQLite divergence: add ``CASE WHEN col IS NULL THEN 1 ELSE 0 END`` to ORDER BY.
    """

    EDGE_NULL = "edge_null"
    """Out-of-range positions produce null (e.g. lag/lead at table edges).

    Applies to: offset.
    """


# ---------------------------------------------------------------------------
# Shared PROPERTY_MAPPING guards and their read-site unwrappers
# ---------------------------------------------------------------------------


def _unwrap_singleton(value: Any) -> Any:
    """The one element of a single-element container, or the value itself.

    The arity rule every scalar guard and unwrapper below shares, in one place. Core unwraps a
    singleton container when it reads a property value (``_unpack_property_value``,
    ``FeatureGroup.resolve_subtype``), so a one-element container is valid caller syntax for one
    value. A multi-element or empty container is returned unchanged, which fails every scalar
    guard's type check: strict membership checks the elements one by one and would otherwise
    wrongly match a composite value such as ``["sum", "max"]``.
    """
    if isinstance(value, (list, tuple, set, frozenset)) and len(value) == 1:
        (element,) = value
        return element
    return value


def option_value(options: Options, key: str, unwrap: Callable[[Any], T]) -> T | None:
    """The option under ``key`` unwrapped from its container; None when the option is absent.

    Absent stays None instead of being routed through an unwrapper (which would turn it into
    the string ``"None"``), so a missing option keeps its family's absent-value behavior.
    """
    value = options.get(key)
    return None if value is None else unwrap(value)


def is_op_token(value: object) -> bool:
    """True for exactly one operation token: a non-empty string, bare or in a single-element container.

    The guard is about arity, not Python syntax: ``("sum",)`` is one token, while ``["sum", "max"]``
    is a composite value that strict membership would otherwise wrongly match element by element.
    """
    token = _unwrap_singleton(value)
    return isinstance(token, str) and bool(token)


def op_token_value(value: object) -> str:
    """The single token of a value is_op_token accepts, unwrapped from its container."""
    return str(_unwrap_singleton(value))


def is_column_ref(value: object) -> bool:
    """True for exactly one column name: a non-empty string, bare or in a single-element container.

    Same value space and arity rule as is_op_token, delegated rather than copied; the second name
    exists for the declaration sites, where a column reference reads nothing like an op token.
    """
    return is_op_token(value)


def column_ref_value(value: object) -> str:
    """The single column name of a value is_column_ref accepts, unwrapped from its container."""
    return op_token_value(value)


def is_scalar_number(value: object) -> bool:
    """True for exactly one int or float, bare or in a single-element container (bool is not a number)."""
    number = _unwrap_singleton(value)
    return isinstance(number, (int, float)) and not isinstance(number, bool)


def is_number_element(value: object) -> bool:
    """Bare element predicate for element_validator slots: core has already unpacked, so no unwrapping (bool is not a number)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def scalar_number_value(value: object) -> int | float:
    """The single number of a value is_scalar_number accepts, unwrapped from its container."""
    number: int | float = _unwrap_singleton(value)
    return number


def _is_feature_ref(value: object) -> bool:
    """One source-feature reference: a non-empty string, or a Feature (duck-typed as core's converter does)."""
    if isinstance(value, str):
        return bool(value)
    return hasattr(value, "options")


def is_in_features_value(value: object) -> bool:
    """True for source-feature references core can resolve: non-empty str or Feature, or a container of those."""
    if isinstance(value, (list, tuple, set, frozenset)):
        # An empty container passes: core's in-feature count check owns it as a plain non-match.
        return all(_is_feature_ref(item) for item in value)
    return _is_feature_ref(value)


def is_positive_int(value: object) -> bool:
    """True only for exactly one positive int, bare or in a single-element container (rejects bool, non-int, n < 1)."""
    n = _unwrap_singleton(value)
    return isinstance(n, int) and not isinstance(n, bool) and n >= 1


def positive_int_value(value: object) -> int:
    """The single int of a value is_positive_int accepts, unwrapped from its container."""
    n: int = _unwrap_singleton(value)
    return n


def always_required(_options: Options) -> bool:
    """required_when predicate for a key required on every path: a PREFIX_PATTERN match otherwise skips the check."""
    return True


#: ASCII decimal >= 1. str.isdigit also accepts superscripts (int() raises) and non-ASCII digits.
_PARAMETRIC_SUFFIX_PATTERN = re.compile(r"[1-9][0-9]*")


def is_parametric_suffix(suffix: str) -> bool:
    """True for the ASCII positive-integer suffix of a parametric operation token (e.g. the 4 in ntile_4)."""
    return _PARAMETRIC_SUFFIX_PATTERN.fullmatch(suffix) is not None


# ---------------------------------------------------------------------------
# Shared rejection-reason reporting
# ---------------------------------------------------------------------------


def _safe_repr(value: Any) -> str:
    """A capped repr that never raises: a hostile __repr__ falls back to the type name."""
    try:
        return reprlib.repr(value)
    except Exception:
        return f"<unreprable {type(value).__name__}>"


class RejectionReasonMixin(FeatureChainParserMixin):
    """Names guard and required_when rejections that core's rejection-reason hook leaves silent."""

    @classmethod
    def _strict_validation_rejection_reason(cls, feature_name: str | FeatureName, options: Options) -> str | None:
        # Core's own hook already gets the value-rejection / name-path-presence / strict-guard
        # precedence right (see its docstring: value rejection first, then name-path presence,
        # then a match_guard rejection on a strict-validation spec), so it is trusted first. Only
        # when it has nothing to report do this mixin's own diagnostics get a turn, for the two
        # gaps core intentionally leaves silent: a required_when miss (core's hook never evaluates
        # required_when at all) and a match_guard rejection on a non-strict spec (core treats that
        # as "this feature group does not match", not "this value is wrong").
        core_reason = super()._strict_validation_rejection_reason(feature_name, options)
        if core_reason is not None:
            return core_reason

        property_mapping = cls._get_property_mapping()
        if property_mapping is None:
            return None

        prefix_patterns = cls._get_prefix_patterns()
        try:
            matched = FeatureChainParser.match_configuration_feature_chain_parser(
                feature_name, options, property_mapping=property_mapping, prefix_patterns=prefix_patterns
            )
        except ValueError:
            # Both match paths validate present option values the same way core's own hook just
            # did above, so a value-rejection ValueError here would already have been returned by
            # super(). A ValueError reaching this point is therefore a parse error (a matched
            # PREFIX_PATTERN with no source feature), which core's own hook also treats as nothing
            # to report.
            return None
        if not matched:
            return None

        # The effective options fold in any name-derived bindings, so a required_when predicate
        # and a match_guard see a name-carried value exactly as the real match path does.
        effective_options = FeatureChainParser.build_effective_options(
            feature_name, prefix_patterns, property_mapping, options
        )
        key = cls._missing_required_when_key(effective_options, property_mapping)
        if key is not None:
            return (
                f"required option '{key}' was not provided; provide it in Options(context=...). "
                f"For a chained name, the child receives only the context keys listed in "
                f"propagate_context_keys"
            )
        # Checked last: the one gap core's own hook leaves silent by design.
        return cls._match_guard_rejection_reason(effective_options, property_mapping)

    @classmethod
    def _missing_required_when_key(
        cls,
        effective_options: Options,
        property_mapping: dict[str, Any],
    ) -> str | None:
        """First key whose required_when predicate fires while the effective options leave it unset."""
        for key, spec in property_mapping.items():
            if not isinstance(spec, PropertySpec):
                continue
            predicate = spec.required_when
            if predicate is None or not callable(predicate):
                continue
            try:
                is_required = bool(predicate(effective_options))
            # Contained exactly like core's own check_required_when (feature_chain_author_guards.py):
            # a predicate that raises cannot judge, so this key is skipped rather than aborting the
            # whole diagnostic.
            except Exception as exc:
                logger.log(
                    contained_raise_log_level(exc),
                    "required_when predicate for '%s' %s; treating it as non-required.",
                    key,
                    contained_raise_reason(exc),
                )
                continue
            # An opted-in explicit None counts as present (#768), same presence test core's own
            # check_required_when uses.
            if is_required and not option_key_is_present(spec, key, effective_options):
                return key
        return None

    @classmethod
    def _match_guard_rejection_reason(cls, options: Options, property_mapping: dict[str, Any]) -> str | None:
        """Formatted reason for the first match_guard rejection of a present value, or None."""
        for key, spec in property_mapping.items():
            if not isinstance(spec, PropertySpec):
                continue
            guard = spec.match_guard
            if guard is None:
                continue
            value = options.get(key)
            if value is None:
                continue
            if cls._guard_accepts(guard, value):
                continue
            # Multi-element with every element accepted, where a flat list is also rejected, is an
            # arity error, so the arity is named. A nested singleton or a container-type rejection
            # falls through as a real guard rejection; the match verdict is a plain non-match either way.
            if (
                isinstance(value, (list, tuple, set, frozenset))
                and len(value) > 1
                and all(cls._guard_accepts(guard, element) for element in value)
                and not cls._guard_accepts(guard, list(value))
            ):
                return f"option '{key}' accepts exactly one value, got {len(value)} elements: {_safe_repr(value)}"
            guard_name = getattr(guard, "__name__", repr(guard))
            return f"match_guard '{guard_name}' for option '{key}' rejected value {_safe_repr(value)}"
        return None

    @staticmethod
    def _guard_accepts(guard: Callable[[Any], Any], value: Any) -> bool:
        """Run one guard, treating a raised TypeError/ValueError/AttributeError as rejection."""
        try:
            return bool(guard(value))
        except (TypeError, ValueError, AttributeError):
            return False
