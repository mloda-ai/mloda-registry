"""Shared match-validation test base for data-operations feature groups.

Provides ``MatchValidationTestBase`` with reusable test methods covering:
- Feature names without a source column prefix
- SQL injection in feature names
- Invalid operation types (pattern-based and options-based)
- Special characters in the operation portion of feature names
- Type confusion via Options (None, int, list)
- Case sensitivity enforcement (lowercase only)

Concrete test classes implement abstract methods to adapt these tests
to each specific operation.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.options import Options


class MatchValidationTestBase:
    """Shared match-validation tests for data-operations feature groups.

    Subclasses implement abstract methods to provide operation-specific
    constants (valid operations, config key, feature name pattern, etc.).
    All concrete test methods are inherited automatically.
    """

    @classmethod
    @abstractmethod
    def feature_group_class(cls) -> Any:
        """Return the base FeatureGroup class under test."""

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
    def options_reject_invalid_types(cls) -> bool:
        """Whether options-based matching rejects invalid operation types.

        False for operations with strict_validation=False on the config key.
        """
        return True

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
            context = {self.config_key(): bad_type, **self.additional_match_options()}
            options = Options(context=context)
            result = self.feature_group_class().match_feature_group_criteria("my_result", options, None)
            assert result is False, f"Should reject via options: {bad_type!r}"

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
        context = {self.config_key(): None, **self.additional_match_options()}
        options = Options(context=context)
        result = self.feature_group_class().match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_integer_type_rejected(self) -> None:
        """An integer as operation type in options must not match."""
        context = {self.config_key(): 42, **self.additional_match_options()}
        options = Options(context=context)
        result = self.feature_group_class().match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_list_type_rejected(self) -> None:
        """A list as operation type in options must not match."""
        valid_ops = list(self.valid_operations())[:2] or ["sum"]
        context = {self.config_key(): valid_ops, **self.additional_match_options()}
        options = Options(context=context)
        result = self.feature_group_class().match_feature_group_criteria("my_result", options, None)
        assert result is False

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
