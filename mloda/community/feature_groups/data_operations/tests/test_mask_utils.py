"""Unit tests for mask_utils shared utilities."""

from __future__ import annotations

import pytest

from mloda.community.feature_groups.data_operations.mask_utils import (
    build_sql_case_when,
    parse_mask_spec,
)


class TestParseMaskSpec:
    def test_none_returns_none(self) -> None:
        assert parse_mask_spec(None) is None

    def test_single_tuple(self) -> None:
        result = parse_mask_spec(("col", "equal", "X"))
        assert result == [("col", "equal", "X")]

    def test_list_of_tuples(self) -> None:
        result = parse_mask_spec([("a", "equal", 1), ("b", "greater_equal", 10)])
        assert result == [("a", "equal", 1), ("b", "greater_equal", 10)]

    def test_invalid_operator(self) -> None:
        with pytest.raises(ValueError, match="Unsupported mask operator"):
            parse_mask_spec(("col", "not_equal", "X"))

    def test_invalid_type(self) -> None:
        with pytest.raises(ValueError, match="must be a tuple or list"):
            parse_mask_spec("bad")

    def test_wrong_tuple_length(self) -> None:
        with pytest.raises(ValueError, match="2 or 3 elements"):
            parse_mask_spec(("a",))

    def test_non_string_column(self) -> None:
        with pytest.raises(ValueError, match="column must be a string"):
            parse_mask_spec((123, "equal", "X"))


class TestBuildSqlCaseWhen:
    def test_single_equal(self) -> None:
        result = build_sql_case_when([("status", "equal", "active")], '"value"')
        assert result == """CASE WHEN "status" = 'active' THEN "value" END"""

    def test_multiple_conditions(self) -> None:
        result = build_sql_case_when(
            [("cat", "equal", "X"), ("val", "greater_equal", 10)],
            '"source"',
        )
        assert '"cat" = ' in result
        assert '"val" >= 10' in result
        assert "AND" in result

    def test_is_in(self) -> None:
        result = build_sql_case_when([("col", "is_in", ["a", "b"])], '"src"')
        assert "IN ('a', 'b')" in result
