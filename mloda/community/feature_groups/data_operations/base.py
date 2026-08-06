"""Shared constants and utilities for all data operation feature groups."""

from __future__ import annotations

import re
import reprlib
from collections.abc import Callable, Container
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import FeatureGroup

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


def is_supported_op_type(value: object, fixed_types: Container[str], parametric_families: tuple[str, ...]) -> bool:
    """True for one of ``fixed_types``, or a ``{family}_{N}`` token of a parametric family with N >= 1."""
    if not isinstance(value, str):
        return False
    if value in fixed_types:
        return True
    return any(
        value.startswith(f"{family}_") and is_parametric_suffix(value[len(family) + 1 :])
        for family in parametric_families
    )


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
        reason = super()._strict_validation_rejection_reason(feature_name, options)
        if reason is not None:
            return reason
        property_mapping = cls._get_property_mapping()
        if property_mapping is None:
            return None
        prefix_patterns = cls._get_prefix_patterns()
        # Both checks only fire for a candidate that would otherwise match; a non-candidate
        # carrying a stray mistyped option stays silent.
        try:
            matched = FeatureChainParser.match_configuration_feature_chain_parser(
                feature_name, options, property_mapping=property_mapping, prefix_patterns=prefix_patterns
            )
        except ValueError:
            matched = False
        if matched:
            # Whether a key is missing goes through cls._validate_required_when so subclass overrides
            # (frame_aggregate's name-path carve-out) win; only the key lookup mirrors the base loop.
            if not cls._validate_required_when(True, feature_name, prefix_patterns, property_mapping, options):
                key = cls._missing_required_when_key(feature_name, prefix_patterns, property_mapping, options)
                if key is not None:
                    return (
                        f"required option '{key}' was not provided; provide it in Options(context=...). "
                        f"For a chained name, the child receives only the context keys listed in "
                        f"propagate_context_keys"
                    )
            guard_reason = cls._match_guard_rejection_reason(options, property_mapping)
            if guard_reason is not None:
                return guard_reason
        return None

    @classmethod
    def _missing_required_when_key(
        cls,
        feature_name: str | FeatureName,
        prefix_patterns: list[str],
        property_mapping: dict[str, Any],
        options: Options,
    ) -> str | None:
        """First key whose required_when predicate fires while the effective options leave it unset."""
        effective_options = cls._build_effective_options(str(feature_name), prefix_patterns, property_mapping, options)
        for key, mapping_entry in property_mapping.items():
            if not isinstance(mapping_entry, dict):
                continue
            predicate = mapping_entry.get(DefaultOptionKeys.required_when)
            if predicate is None or not callable(predicate):
                continue
            if predicate(effective_options) and effective_options.get(key) is None:
                return key
        return None

    @classmethod
    def _match_guard_rejection_reason(cls, options: Options, property_mapping: dict[str, Any]) -> str | None:
        """Formatted reason for the first match_guard rejection of a present value, or None."""
        for key, mapping_entry in property_mapping.items():
            if not isinstance(mapping_entry, dict):
                continue
            guard = mapping_entry.get(DefaultOptionKeys.match_guard)
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


# ---------------------------------------------------------------------------
# Shared source-column and operation-type plumbing
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    # Borrowed for type checking only. Inheriting the chain-parser API at runtime would move
    # FeatureChainParserMixin and FeatureGroup in every family's __mro__, and core reads that
    # index as link specificity (resolve_links._inheritance_distance).
    _ChainParserApi = FeatureChainParserMixin
else:
    _ChainParserApi = object


def _own_names(cls: type) -> frozenset[str]:
    """Every non-dunder name the class itself defines."""
    return frozenset(name for name in vars(cls) if not name.startswith("__"))


#: Names core resolves BEFORE an appended mixin at runtime, so a mixin defining one would be dead code.
_CORE_OWNED_NAMES = frozenset(
    name for klass in (*FeatureChainParserMixin.__mro__, *FeatureGroup.__mro__) for name in _own_names(klass)
)


class _MixinContract(_ChainParserApi):
    """Class-creation guard for the accessor mixins below: what mypy --strict cannot see.

    ``_ChainParserApi`` is FeatureChainParserMixin while type checking and ``object`` at runtime, so a
    type checker puts these mixins AHEAD of core in the ``__mro__`` while CPython puts them BEHIND it.
    A mixin name core also owns therefore type-checks as the winner while being dead at runtime; a bare
    annotation (``PARTITION_BY: str``) is a declaration to mypy that does not exist at runtime; and a
    mixin not listed last moves the ancestors behind it. Each raises here, naming class and attribute.
    """

    #: Names a mixin reads off the mixing class without defining them itself.
    REQUIRED_ATTRS: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__module__ == __name__:
            shadowed = sorted(_own_names(cls) & _CORE_OWNED_NAMES)
            if shadowed:
                raise TypeError(f"{cls.__name__} defines {shadowed[0]!r}, which core owns and resolves first")
            return
        appended = tuple(
            base for base in cls.__bases__ if base.__module__ == __name__ and issubclass(base, _MixinContract)
        )
        if appended and cls.__bases__[-len(appended) :] != appended:
            raise TypeError(
                f"{cls.__name__} must list {appended[0].__name__} last in its bases, so every other "
                f"ancestor keeps its __mro__ index (core reads that index as link specificity)"
            )
        declared = frozenset(name for base in cls.__mro__ if base.__module__ != __name__ for name in _own_names(base))
        for mixin in cls.__mro__:
            if mixin.__module__ != __name__:
                continue
            for attr in vars(mixin).get("REQUIRED_ATTRS", ()):
                if attr not in declared:
                    raise TypeError(f"{cls.__name__} must declare {attr!r}, which {mixin.__name__} reads at runtime")


class SingleSourceMixin(_MixinContract):
    """One source column per feature: the name-or-in_features read and its arity errors.

    Mix in LAST so appending leaves every existing ancestor at its current ``__mro__`` index (see
    ``_ChainParserApi`` above). Core owns ``_extract_source_features`` and ``input_features`` and
    resolves both BEFORE an appended mixin, so a family delegates those two explicitly.
    """

    #: Family name rendered in the arity and ordering errors; None renders the class name instead.
    SOURCE_LABEL: ClassVar[str | None] = None

    #: Reject an empty ``in_features`` instead of returning an empty source list.
    ENFORCE_MIN_IN_FEATURES: ClassVar[bool] = False

    #: Report more than MAX_IN_FEATURES sources as a family arity error instead of leaving it to core.
    ENFORCE_MAX_IN_FEATURES: ClassVar[bool] = False

    #: Run core's in-feature count check on the ``in_features`` branch of ``_single_source_input_features``.
    VALIDATE_IN_FEATURE_COUNT: ClassVar[bool] = True

    @classmethod
    def _source_label(cls) -> str:
        """The noun the arity and ordering errors name the family with."""
        return cls.SOURCE_LABEL or cls.__name__

    @classmethod
    def _source_from_name(cls, feature_name: str) -> str | None:
        """The source column encoded in the feature name, or None when the name encodes none."""
        _operation_config, source_feature = FeatureChainParser.parse_feature_name(
            feature_name, cls._get_prefix_patterns()
        )
        # The parser returns both halves or neither (a match without a source raises), so an
        # empty source is always the no-match case.
        return source_feature or None

    @classmethod
    def _single_source_features(cls, feature: Feature) -> list[str]:
        """The one source column, from the feature name when it encodes one, else from ``in_features``."""
        source_feature = cls._source_from_name(feature.name)
        if source_feature is not None:
            return [source_feature]

        source_names = [str(f.name) for f in feature.options.get_in_features()]
        cls._validate_source_arity(source_names)
        return source_names

    @classmethod
    def _validate_source_arity(cls, source_names: list[str]) -> None:
        """Reject an ``in_features`` list outside the family's arity."""
        if cls.ENFORCE_MIN_IN_FEATURES and len(source_names) < cls.MIN_IN_FEATURES:
            raise ValueError(
                f"{cls._source_label()} requires at least {cls.MIN_IN_FEATURES} source feature, "
                f"but got {len(source_names)} (in_features is empty)."
            )
        cls._reject_extra_sources(source_names)

    @classmethod
    def _reject_extra_sources(cls, source_names: list[str]) -> None:
        """Reject more than MAX_IN_FEATURES source columns."""
        maximum = cls.MAX_IN_FEATURES
        if not cls.ENFORCE_MAX_IN_FEATURES or maximum is None or len(source_names) <= maximum:
            return
        raise ValueError(
            f"{cls._source_label()} supports at most {maximum} source feature, "
            f"but got {len(source_names)}: {source_names}"
        )

    def _single_source_input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        """The one input Feature the name encodes, else the ``in_features`` set."""
        source_feature = self._source_from_name(str(feature_name))
        if source_feature is not None:
            return {Feature(source_feature)}

        in_features_set = options.get_in_features()
        if self.VALIDATE_IN_FEATURE_COUNT:
            self._validate_in_feature_count(list(in_features_set), str(feature_name))
        return set(in_features_set)


class PartitionedSourceMixin(SingleSourceMixin):
    """SingleSourceMixin plus the PARTITION_BY read, split out so no family inherits an accessor it cannot honour."""

    REQUIRED_ATTRS = ("PARTITION_BY",)

    PARTITION_BY: str

    @classmethod
    def _extract_partition_by(cls, feature: Feature) -> list[str]:
        """Return ``partition_by`` as a list (defaulting to ``[]`` when absent)."""
        partition_by = feature.options.get(cls.PARTITION_BY)
        if partition_by is None:
            return []
        return list(partition_by)


class OrderedSourceMixin(PartitionedSourceMixin):
    """PartitionedSourceMixin plus the ORDER_BY read, for the families that declare both keys."""

    REQUIRED_ATTRS = ("ORDER_BY",)

    ORDER_BY: str

    #: Order by the source column when ``order_by`` is absent; False makes the key required.
    ORDER_BY_DEFAULTS_TO_SOURCE: ClassVar[bool] = False

    @classmethod
    def _extract_order_by(cls, feature: Feature, source_col: str | None = None) -> str:
        """Return ``order_by``, falling back to ``source_col`` only for a family whose contract defaults to it."""
        order_by = option_value(feature.options, cls.ORDER_BY, column_ref_value)
        if order_by is not None:
            return order_by
        if cls.ORDER_BY_DEFAULTS_TO_SOURCE and source_col is not None:
            return source_col
        raise ValueError(f"{cls._source_label()} requires an 'order_by' column in Options context.")


class OpTypeAccessorMixin(_MixinContract):
    """One operation-type token per feature: the name-first read, its options fallback, and their errors.

    Mix in LAST, like SingleSourceMixin. Families keep their own accessor names as delegators, so a
    subclass override of e.g. ``_resolve_agg_type`` still wins at every call site.
    """

    REQUIRED_ATTRS = ("_op_type_key", "OP_TYPE_LABEL")

    #: Noun rendered in the extraction errors (e.g. ``"rank type"``).
    OP_TYPE_LABEL: ClassVar[str]

    @classmethod
    def _op_type_key(cls) -> str:
        """The option key the token falls back to; a family returns its own constant so overriding it stays live."""
        raise NotImplementedError(f"{cls.__name__} must return its operation-type option key from _op_type_key")

    @classmethod
    def _extract_op_type(cls, feature_name: str, options: Options | None = None) -> str:
        """The type the name encodes, else the one in ``options`` when a caller passes them; raises on neither."""
        operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, cls._get_prefix_patterns())
        if operation_config is not None:
            return operation_config
        if options is None:
            raise ValueError(f"Could not extract {cls.OP_TYPE_LABEL} from feature name: {feature_name}")
        op_type = options.get(cls._op_type_key())
        if op_type is None:
            raise ValueError(f"Could not extract {cls.OP_TYPE_LABEL} for {feature_name}")
        return op_token_value(op_type)

    @classmethod
    def _resolve_op_type(cls, feature_name: str, options: Options) -> str | None:
        """The same read as _extract_op_type, with every miss (an unparsable name included) reported as None."""
        try:
            operation_config, _ = FeatureChainParser.parse_feature_name(feature_name, cls._get_prefix_patterns())
        except ValueError:
            return None
        if operation_config is not None:
            return operation_config
        op_type = options.get(cls._op_type_key())
        return None if op_type is None else op_token_value(op_type)
