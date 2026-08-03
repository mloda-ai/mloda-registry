"""Shared match-validation and scalar-arity test bases for data-operations feature groups.

``ScalarArityTestBase`` owns the arity contract of every scalar PROPERTY_MAPPING key a
family declares. Core unwraps a one-element container when it reads a property value, so
a single-element container is valid caller syntax for one value: it must produce the same
match verdict, switch the same conditional requirements, reach dispatch as the same value
and compute the same column as its bare form, while a multi-element container, a value of
the wrong type and a value outside the key's value space must stay rejected at every
arity. Families declare their keys as ``token_cases`` and the harness derives the checks;
nothing here needs an operation config key, so families that carry the operation entirely
in the feature name (ema's span, ffill's fixed suffix, sessionization's threshold) use
this base on its own.

``MatchValidationTestBase`` builds on it for families that do declare an operation config
key, adding:
- Feature names without a source column prefix
- SQL injection in feature names
- Invalid operation types (pattern-based and options-based)
- Special characters in the operation portion of feature names
- Type confusion via Options (None, int)
- Case sensitivity enforcement (lowercase only)
- Declaration/dispatch drift between the name path and the config path: the
  acceptance half is opt-out (``parity_operations``), the rejection half is
  opt-in (``malformed_operations``)

Concrete test classes implement abstract methods to adapt these tests
to each specific operation.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import DefaultOptionKeys


def _is_container(value: Any) -> bool:
    """True for the sequence containers core unpacks element-wise; a str is one scalar, not a sequence."""
    return isinstance(value, (list, tuple, set, frozenset))


#: Sentinel for "derive the wrong-type value from the token"; ``None`` opts a case out.
_DERIVED = object()


@dataclass(frozen=True)
class TokenCase:
    """One state of one scalar config key, declared for the single-element checks.

    ``context`` adds to and ``without`` drops from ``base_context()``, which is how a
    state that turns a conditional requirement on or off is declared (a sized frame type
    without its ``frame_size``, ``first`` without its ``order_by``).
    ``matches`` is the verdict that state must produce, bare and wrapped alike.

    ``other`` is a second valid value for the same key, used for the multi-element
    rejection; ``None`` skips it. Any non-container token takes part in the
    single-element checks: an operation token, a column reference and a number are
    all read back as one value. A token that is already a container is skipped,
    since wrapping it again would change its arity rather than its syntax.

    The remaining fields declare, rather than hand-roll, the per-key checks:

    ``wrong_type`` is a value of a type the key never accepts, rejected bare and wrapped
    alike; it defaults to the opposite kind of the token (an int under a string key, a
    string under a numeric one) and ``None`` opts the case out.
    ``invalid`` holds values inside the key's type but outside its value space (``0``
    bins, a percentile of ``1.5``, an unparsable op token); unwrapping is syntax, not
    permission, so each must stay rejected bare and wrapped alike.
    ``required`` marks a key whose absence the dispatch path must reject with a
    ``ValueError`` naming the key, rather than silently substituting a default.
    ``compute`` opts a state out of the end-to-end compute check for families that
    declare ``compute_values``.
    """

    key: str
    token: Any
    other: Any = None
    context: dict[str, Any] = field(default_factory=dict)
    without: tuple[str, ...] = ()
    matches: bool = True
    wrong_type: Any = _DERIVED
    invalid: tuple[Any, ...] = ()
    required: bool = False
    compute: bool = True

    def wrong_type_value(self) -> Any:
        """A value of a type this key never accepts, or None when the case opted out."""
        if self.wrong_type is not _DERIVED:
            return self.wrong_type
        return 123 if isinstance(self.token, str) else "five"


class ScalarArityTestBase:
    """Shared arity tests for the scalar PROPERTY_MAPPING keys a data-operations family declares.

    Subclasses declare their keys in ``token_cases`` and, where the family has them, the
    extractors (``dispatch_values``) and the backend call (``compute_values``) a wrapped
    value has to survive. Everything else is derived.
    """

    @classmethod
    @abstractmethod
    def feature_group_class(cls) -> Any:
        """Return the base FeatureGroup class under test."""

    @classmethod
    def match_class(cls) -> Any:
        """The class whose matcher the arity checks call.

        Defaults to the feature group itself. Families whose matcher only resolves on a
        compute-framework subclass return that subclass instead.
        """
        return cls.feature_group_class()

    @classmethod
    def match_feature_name(cls) -> str:
        """The feature name the arity checks match against.

        ``"my_result"`` for families that carry the operation in the config; families
        that carry it in the feature name return a name holding that operation.
        """
        return "my_result"

    @classmethod
    def base_context(cls) -> dict[str, Any]:
        """The options context every arity case starts from, before the key under test."""
        return {}

    @classmethod
    def token_cases(cls) -> list[TokenCase]:
        """States whose value must behave the same bare and in a single-element container.

        Core unwraps a one-element container when it reads a property value, so
        ``("sum",)`` and ``(5,)`` are valid caller syntax for one value: each must produce
        the same verdict as its bare form and reach dispatch as ``"sum"`` / ``5``, not as
        ``"('sum',)"`` / ``(5,)``.

        Empty by default. Declare one case per scalar key the family reads, plus the
        states where a value turns a conditional requirement on or off.
        """
        return []

    @classmethod
    def dispatch_values(cls, options: Options) -> list[Any]:
        """Values the dispatch path resolves from ``options``.

        Empty by default, which checks the match verdict only. Override with the
        family's extractors to also assert that a wrapped value reaches dispatch
        unwrapped, rather than as the string form of its container.
        """
        return []

    @classmethod
    def compute_values(cls, options: Options) -> list[Any] | None:
        """The column the backend computes for ``options``, or None to skip the compute check.

        None by default. Override for families where a wrapped value used to survive
        discovery and then fail (or silently mis-sort) inside ``calculate_feature``;
        ``make_feature_set`` and ``extract_column`` in ``helpers`` build the call.
        """
        return None

    # -- Case options ---------------------------------------------------------

    def _case_context(self, case: TokenCase) -> dict[str, Any]:
        """The context of ``case`` without the key under test."""
        context: dict[str, Any] = dict(self.base_context())
        context.update(case.context)
        for key in case.without:
            context.pop(key, None)
        context.pop(case.key, None)
        return context

    def _case_options(self, case: TokenCase, value: Any) -> Options:
        """Options for ``case`` with its key holding ``value``."""
        return Options(context={**self._case_context(case), case.key: value})

    def _match(self, options: Options) -> bool:
        """Match verdict of the configuration-based path for the given options."""
        matcher = self.match_class().match_feature_group_criteria
        return bool(matcher(self.match_feature_name(), options, None))

    def _scalar_cases(self) -> list[TokenCase]:
        """The declared cases whose token is one scalar value rather than a container."""
        return [case for case in self.token_cases() if not _is_container(case.token)]

    @classmethod
    def _required_when_predicates(cls) -> dict[str, Any]:
        """The conditional-requirement predicates the feature group declares in PROPERTY_MAPPING."""
        mapping = getattr(cls.feature_group_class(), "PROPERTY_MAPPING", None) or {}
        predicates = {}
        for key, entry in mapping.items():
            predicate = entry.get(DefaultOptionKeys.required_when) if isinstance(entry, dict) else None
            if callable(predicate):
                predicates[key] = predicate
        return predicates

    def _dispatch_or_none(self, options: Options) -> list[Any] | None:
        """Values the dispatch path resolves, or None for a state its extractors reject."""
        try:
            return self.dispatch_values(options)
        except ValueError:
            return None

    # -- Single-element containers -------------------------------------------

    def test_single_element_container_matches(self) -> None:
        """A bare value and its single-element containers must match alike and dispatch alike."""
        cases = self._scalar_cases()
        if not cases:
            pytest.skip("no scalar value declared for the single-element checks")
        for case in cases:
            bare = self._case_options(case, case.token)
            assert self._match(bare) is case.matches, f"{case.key}={case.token!r} should match: {case.matches}"
            expected = self._dispatch_or_none(bare)
            for value in ((case.token,), [case.token]):
                options = self._case_options(case, value)
                assert self._match(options) is case.matches, f"{case.key}={value!r} should match: {case.matches}"
                assert self._dispatch_or_none(options) == expected, (
                    f"{case.key}={value!r} should dispatch as {case.token!r}"
                )

    def test_single_element_container_preserves_requirements(self) -> None:
        """A wrapped value must switch the same conditional requirements on as a bare one.

        ``required_when`` predicates read the option raw, so this is where a container
        that never gets unwrapped drops a requirement and lets an under-specified
        feature match at discovery and fail at compute.
        """
        predicates = self._required_when_predicates()
        cases = self._scalar_cases()
        if not predicates or not cases:
            pytest.skip("no required_when predicate keyed off a scalar value")
        for case in cases:
            bare = self._case_options(case, case.token)
            expected = {key: bool(predicate(bare)) for key, predicate in predicates.items()}
            for value in ((case.token,), [case.token]):
                options = self._case_options(case, value)
                resolved = {key: bool(predicate(options)) for key, predicate in predicates.items()}
                assert resolved == expected, f"{case.key}={value!r} should require what {case.token!r} requires"

    def test_single_element_container_computes_like_bare(self) -> None:
        """A wrapped value must reach the backend as the value, not as its container's string form.

        The match verdict and the extractors can both be right while the backend still
        receives the container, which is how ``constant=(5,)`` matched at discovery and
        then raised ``must be int or float, got tuple`` inside calculate_feature (#339).
        """
        cases = [case for case in self._scalar_cases() if case.matches and case.compute]
        if not cases:
            pytest.skip("no computable scalar value declared")
        for case in cases:
            expected = self.compute_values(self._case_options(case, case.token))
            if expected is None:
                pytest.skip("family declares no compute check")
            for value in ((case.token,), [case.token]):
                assert self.compute_values(self._case_options(case, value)) == expected, (
                    f"{case.key}={value!r} should compute like {case.token!r}"
                )

    def test_multi_element_container_rejected(self) -> None:
        """Two operations in one container are not one operation, whatever the container type."""
        cases = [case for case in self.token_cases() if case.other is not None and case.matches]
        if not cases:
            pytest.skip("no token key declares a second operation")
        for case in cases:
            for value in ([case.token, case.other], (case.token, case.other)):
                assert self._match(self._case_options(case, value)) is False, (
                    f"Config path should reject {case.key}={value!r}"
                )

    # -- Value space ----------------------------------------------------------

    def test_wrong_type_rejected_at_every_arity(self) -> None:
        """A value of the wrong type for a scalar key is a non-match, bare or wrapped."""
        cases = [case for case in self._scalar_cases() if case.matches and case.wrong_type_value() is not None]
        if not cases:
            pytest.skip("no scalar key declares a wrong-type value")
        for case in cases:
            wrong = case.wrong_type_value()
            for value in (wrong, (wrong,), [wrong]):
                assert self._match(self._case_options(case, value)) is False, (
                    f"{case.key}={value!r} is the wrong type and should not match"
                )

    def test_invalid_value_rejected_at_every_arity(self) -> None:
        """Unwrapping is syntax, not permission: a value the key rejects stays rejected wrapped."""
        cases = [case for case in self._scalar_cases() if case.invalid]
        if not cases:
            pytest.skip("no scalar key declares a value outside its value space")
        for case in cases:
            for invalid in case.invalid:
                for value in (invalid, (invalid,), [invalid]):
                    assert self._match(self._case_options(case, value)) is False, (
                        f"{case.key}={value!r} is outside the key's value space and should not match"
                    )

    def test_missing_required_key_raises(self) -> None:
        """A key the family requires must fail at dispatch naming itself, not fall back to a default."""
        cases = [case for case in self.token_cases() if case.required]
        if not cases:
            pytest.skip("no scalar key is declared required")
        for case in cases:
            with pytest.raises(ValueError, match=case.key):
                self.dispatch_values(Options(context=self._case_context(case)))


class MatchValidationTestBase(ScalarArityTestBase):
    """Shared match-validation tests for data-operations feature groups.

    Subclasses implement abstract methods to provide operation-specific
    constants (valid operations, config key, feature name pattern, etc.).
    All concrete test methods are inherited automatically.
    """

    @classmethod
    @abstractmethod
    def valid_operations(cls) -> set[str]:
        """Return the canonical set of valid operation strings."""

    @classmethod
    @abstractmethod
    def config_key(cls) -> str:
        """Return the options context key (e.g. 'aggregation_type')."""

    @classmethod
    @abstractmethod
    def build_feature_name(cls, operation: str) -> str:
        """Build a feature name for the given operation.

        Should produce a name that would match if the operation were valid.
        """

    @classmethod
    @abstractmethod
    def build_feature_name_no_source(cls) -> str:
        """Build a feature name with the right op/suffix but no source column prefix.

        For example, ``"sum_aggr"`` instead of ``"value_int__sum_aggr"``.
        """

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        """Additional options needed for options-based matching.

        Override to provide required keys like ``in_features``,
        ``partition_by``, or ``order_by``.
        """
        return {}

    @classmethod
    def pattern_match_options(cls) -> Options:
        """Options to use alongside pattern-based feature name tests.

        Override for operations that require context even for pattern-based
        matching (e.g. frame_aggregate requires partition_by and order_by).
        """
        return Options()

    @classmethod
    def config_value(cls, operation: str) -> Any:
        """Map a name-path operation token to the value the config path expects.

        Identity by default; override for families whose config vocabulary
        differs from the feature name token (e.g. percentile's float).
        """
        return operation

    @classmethod
    def base_context(cls) -> dict[str, Any]:
        """The operation key holding a valid value, plus whatever else matching requires."""
        return {cls.config_key(): cls._primary_value(), **cls.additional_match_options()}

    @classmethod
    def token_cases(cls) -> list[TokenCase]:
        """States whose value must behave the same bare and in a single-element container.

        Defaults to the primary operation key holding parity operations. Override to
        append the other scalar keys a family dispatches on, and the states where a value
        turns a conditional requirement on or off.
        """
        operations = sorted(cls.parity_operations())
        if not operations:
            return []
        values = [cls.config_value(operation) for operation in operations[:2]]
        return [TokenCase(cls.config_key(), values[0], values[1] if len(values) > 1 else None)]

    @classmethod
    def options_reject_invalid_types(cls) -> bool:
        """Whether options-based matching rejects invalid operation types.

        False for operations with strict_validation=False on the config key.
        """
        return True

    @classmethod
    def parity_operations(cls) -> set[str]:
        """Operations that both the name path and the config path must accept.

        Defaults to ``valid_operations()``; the drift check is opt-out, so
        override only to narrow the set or to exempt an operation.
        """
        return cls.valid_operations()

    @classmethod
    def malformed_operations(cls) -> set[str]:
        """Operations that both the name path and the config path must reject.

        Empty by default; override to opt in to the drift check.
        """
        return set()

    @classmethod
    def _primary_value(cls) -> Any:
        """The value the primary operation key holds while another key is under test."""
        return cls.config_value(sorted(cls.parity_operations())[0])

    def _config_options(self, value: Any) -> Options:
        """Options carrying ``value`` as the operation on the configuration path."""
        return Options(context={self.config_key(): value, **self.additional_match_options()})

    # -- No source column ------------------------------------------------------

    def test_no_match_no_source_column(self) -> None:
        """Feature name without a source column prefix must not match."""
        name = self.build_feature_name_no_source()
        options = self.pattern_match_options()
        result = self.feature_group_class().match_feature_group_criteria(name, options, None)
        assert result is False, f"Should reject feature name without source column: {name}"

    # -- SQL injection -------------------------------------------------------

    SQL_INJECTION_SUFFIXES = [
        "; DROP TABLE users",
        "' OR '1'='1",
        "); DELETE FROM data--",
    ]

    def test_sql_injection_rejected_by_match(self) -> None:
        """SQL injection payloads appended to feature names must be rejected."""
        base = self.build_feature_name(next(iter(self.valid_operations())))
        options = self.pattern_match_options()
        for suffix in self.SQL_INJECTION_SUFFIXES:
            malicious = f"{base}{suffix}"
            result = self.feature_group_class().match_feature_group_criteria(malicious, options, None)
            assert result is False, f"Should reject: {malicious}"

    # -- Invalid operation types ---------------------------------------------

    INVALID_TYPES = ["drop_table", "exec", "eval", "__import__", ""]

    def test_invalid_type_rejected_by_pattern_match(self) -> None:
        """Feature names with invalid operation types must not match."""
        options = self.pattern_match_options()
        for bad_type in self.INVALID_TYPES:
            feature_name = self.build_feature_name(bad_type)
            result = self.feature_group_class().match_feature_group_criteria(feature_name, options, None)
            assert result is False, f"Should reject operation type: {bad_type!r}"

    def test_invalid_type_rejected_by_options_match(self) -> None:
        """Options-based configuration with invalid types must not match."""
        if not self.options_reject_invalid_types():
            pytest.skip("strict_validation is False for this operation's config key")
        for bad_type in self.INVALID_TYPES:
            if bad_type == "":
                continue
            assert self._match(self._config_options(bad_type)) is False, f"Should reject via options: {bad_type!r}"

    # -- Special characters --------------------------------------------------

    def test_special_chars_in_operation_rejected(self) -> None:
        """Feature names with special characters in the operation are rejected."""
        valid_op = next(iter(self.valid_operations()))
        if len(valid_op) < 2:
            return
        mid = len(valid_op) // 2
        options = self.pattern_match_options()
        for char in ["'", '"', ";", ")", "--"]:
            mangled = valid_op[:mid] + char + valid_op[mid:]
            feature_name = self.build_feature_name(mangled)
            result = self.feature_group_class().match_feature_group_criteria(feature_name, options, None)
            assert result is False, f"Should reject special char {char!r} in: {feature_name}"

    # -- Type confusion ------------------------------------------------------

    def test_none_type_rejected(self) -> None:
        """None as operation type in options must not match."""
        assert self._match(self._config_options(None)) is False

    def test_integer_type_rejected(self) -> None:
        """An integer as operation type in options must not match."""
        assert self._match(self._config_options(42)) is False

    # -- Case sensitivity ----------------------------------------------------

    def test_uppercase_rejected(self) -> None:
        """Uppercase operation types must be rejected."""
        options = self.pattern_match_options()
        for op in self.valid_operations():
            upper = op.upper()
            if upper == op:
                continue
            feature_name = self.build_feature_name(upper)
            result = self.feature_group_class().match_feature_group_criteria(feature_name, options, None)
            assert result is False, f"Should reject uppercase: {upper}"

    def test_mixed_case_rejected(self) -> None:
        """Mixed-case operation types must be rejected."""
        options = self.pattern_match_options()
        for op in self.valid_operations():
            mixed = op.capitalize()
            if mixed == op:
                continue
            feature_name = self.build_feature_name(mixed)
            result = self.feature_group_class().match_feature_group_criteria(feature_name, options, None)
            assert result is False, f"Should reject mixed case: {mixed}"

    # -- Declaration / dispatch drift ----------------------------------------

    def _match_by_name(self, operation: str) -> bool:
        """Match verdict of the name-based path for the given operation."""
        feature_name = self.build_feature_name(operation)
        result = self.feature_group_class().match_feature_group_criteria(
            feature_name, self.pattern_match_options(), None
        )
        return bool(result)

    def _match_by_config(self, operation: str) -> bool:
        """Match verdict of the configuration-based path for the given operation."""
        return self._match(self._config_options(self.config_value(operation)))

    def test_operations_match_on_both_paths(self) -> None:
        """Operations accepted by one path must be accepted by the other."""
        operations = self.parity_operations()
        if not operations:
            pytest.skip("no parity_operations declared for this operation")
        for operation in sorted(operations):
            assert self._match_by_name(operation) is True, f"Name path should accept: {operation!r}"
            assert self._match_by_config(operation) is True, f"Config path should accept: {operation!r}"

    def test_malformed_operations_rejected_on_both_paths(self) -> None:
        """Operations rejected by one path must be rejected by the other."""
        operations = self.malformed_operations()
        if not operations:
            pytest.skip("no malformed_operations declared for this operation")
        for operation in sorted(operations):
            assert self._match_by_name(operation) is False, f"Name path should reject: {operation!r}"
            assert self._match_by_config(operation) is False, f"Config path should reject: {operation!r}"
