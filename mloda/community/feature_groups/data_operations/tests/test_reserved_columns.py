"""Tests for the reserved-helper-column guard."""

from __future__ import annotations

import pytest

from mloda.community.feature_groups.data_operations.reserved_columns import (
    RESERVED_PREFIX,
    assert_no_reserved_columns,
)


class TestReservedPrefix:
    def test_prefix_is_double_underscore_mloda_underscore(self) -> None:
        """The reserved prefix is ``__mloda_``. Tests and docs rely on this literal."""
        assert RESERVED_PREFIX == "__mloda_"


class TestAssertNoReservedColumns:
    def test_no_collision_is_silent(self) -> None:
        """User columns that do not start with the prefix are accepted."""
        assert_no_reserved_columns(["region", "value", "timestamp"])

    def test_empty_columns_is_silent(self) -> None:
        """Empty input does not trigger the guard."""
        assert_no_reserved_columns([])

    def test_non_matching_prefix_is_silent(self) -> None:
        """Columns that merely contain ``mloda`` but do not *start* with the
        prefix are accepted (the guard is prefix-based, not substring-based)."""
        assert_no_reserved_columns(["my_mloda_col", "_mloda_col", "mloda_rn"])

    def test_collision_raises_value_error(self) -> None:
        """A column starting with the reserved prefix raises ``ValueError``."""
        with pytest.raises(ValueError):
            assert_no_reserved_columns(["region", "__mloda_rn__"])

    def test_message_quotes_collided_name(self) -> None:
        """The colliding column name appears in the message, via ``repr()``."""
        with pytest.raises(ValueError, match="'__mloda_rn__'"):
            assert_no_reserved_columns(["__mloda_rn__"])

    def test_message_lists_multiple_collisions_sorted(self) -> None:
        """Multiple colliding names appear alphabetically sorted in the message."""
        with pytest.raises(ValueError, match="'__mloda_a__', '__mloda_b__'"):
            assert_no_reserved_columns(["__mloda_b__", "__mloda_a__"])

    def test_message_uses_plural_suffix_when_multiple(self) -> None:
        """The message reads ``column names`` (plural) when >1 collision."""
        with pytest.raises(ValueError, match="reserved helper column names"):
            assert_no_reserved_columns(["__mloda_a__", "__mloda_b__"])

    def test_message_uses_singular_when_one(self) -> None:
        """The message reads ``column name`` (singular) when exactly 1 collision."""
        with pytest.raises(ValueError, match="reserved helper column name[:]"):
            assert_no_reserved_columns(["__mloda_only__"])

    def test_message_mentions_framework(self) -> None:
        """``framework`` kwarg is surfaced in the error message when provided."""
        with pytest.raises(ValueError, match="for DuckDB"):
            assert_no_reserved_columns(["__mloda_rn__"], framework="DuckDB")

    def test_message_mentions_operation(self) -> None:
        """``operation`` kwarg is surfaced in the error message when provided."""
        with pytest.raises(ValueError, match="frame aggregate"):
            assert_no_reserved_columns(
                ["__mloda_rn__"],
                framework="SQLite",
                operation="frame aggregate",
            )

    def test_message_includes_remediation(self) -> None:
        """The error tells the user how to fix the collision."""
        with pytest.raises(ValueError, match="rename the input column"):
            assert_no_reserved_columns(["__mloda_rn__"])

    def test_accepts_arbitrary_iterable(self) -> None:
        """The helper takes any iterable of strings (list, set, generator, keys view)."""

        def _gen() -> list[str]:
            return ["region", "__mloda_rn__"]

        with pytest.raises(ValueError):
            assert_no_reserved_columns(iter(_gen()))

    def test_deduplicates_identical_collisions(self) -> None:
        """Identical colliding names are not repeated in the message."""
        with pytest.raises(ValueError) as exc:
            assert_no_reserved_columns(["__mloda_rn__", "__mloda_rn__"])
        # Only one occurrence of the name in the message.
        assert str(exc.value).count("'__mloda_rn__'") == 1

    def test_exact_prefix_is_reserved(self) -> None:
        """A column named exactly the prefix (``__mloda_``) collides."""
        with pytest.raises(ValueError):
            assert_no_reserved_columns(["__mloda_"])

    def test_uppercase_prefix_is_reserved(self) -> None:
        """An all-uppercase variant collides because SQLite and DuckDB
        unquoted identifiers are case-insensitive."""
        with pytest.raises(ValueError):
            assert_no_reserved_columns(["__MLODA_RN__"])

    def test_mixed_case_prefix_is_reserved(self) -> None:
        """A mixed-case variant collides for the same case-insensitivity reason."""
        with pytest.raises(ValueError):
            assert_no_reserved_columns(["__Mloda_Rn__"])

    def test_exact_uppercase_prefix_is_reserved(self) -> None:
        """The exact prefix in uppercase (``__MLODA_``) collides."""
        with pytest.raises(ValueError):
            assert_no_reserved_columns(["__MLODA_"])

    def test_message_preserves_original_casing(self) -> None:
        """The colliding name in the error message keeps the user-supplied casing."""
        with pytest.raises(ValueError, match="'__MLODA_RN__'"):
            assert_no_reserved_columns(["__MLODA_RN__"])

    def test_prefix_boundary_preserved_case_insensitively(self) -> None:
        """A name that shares letters but lacks the trailing ``_`` after ``mloda``
        is still accepted (the prefix boundary is enforced case-insensitively)."""
        assert_no_reserved_columns(["__MLODAX_foo"])
