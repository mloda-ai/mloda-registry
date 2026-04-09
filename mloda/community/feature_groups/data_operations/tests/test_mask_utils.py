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

    def test_is_in_string_rejected(self) -> None:
        with pytest.raises(ValueError, match="is_in values must be a list"):
            parse_mask_spec(("col", "is_in", "DE"))

    def test_is_in_bytes_rejected(self) -> None:
        with pytest.raises(ValueError, match="is_in values must be a list"):
            parse_mask_spec(("col", "is_in", b"DE"))

    def test_is_in_list_accepted(self) -> None:
        result = parse_mask_spec(("col", "is_in", ["A", "B"]))
        assert result == [("col", "is_in", ["A", "B"])]

    def test_is_in_set_accepted(self) -> None:
        result = parse_mask_spec(("col", "is_in", {"A", "B"}))
        assert result is not None
        assert result[0][1] == "is_in"

    def test_unsupported_value_type_rejected(self) -> None:
        from datetime import datetime

        with pytest.raises(ValueError, match="Mask value must be"):
            parse_mask_spec(("col", "equal", datetime.now()))

    def test_greater_than_operator(self) -> None:
        result = parse_mask_spec(("col", "greater_than", 10))
        assert result == [("col", "greater_than", 10)]

    def test_two_element_equal_none(self) -> None:
        """2-element tuple sets val=None, which is valid for 'equal'."""
        result = parse_mask_spec(("col", "equal"))
        assert result == [("col", "equal", None)]

    def test_is_in_empty_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            parse_mask_spec(("col", "is_in", []))

    def test_is_in_empty_set_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            parse_mask_spec(("col", "is_in", set()))

    def test_two_element_greater_than_rejected(self) -> None:
        """2-element tuple is only valid for 'equal', not other operators."""
        with pytest.raises(ValueError, match="only valid for 'equal'"):
            parse_mask_spec(("col", "greater_than"))

    def test_two_element_is_in_rejected(self) -> None:
        with pytest.raises(ValueError, match="only valid for 'equal'"):
            parse_mask_spec(("col", "is_in"))


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

    def test_greater_than(self) -> None:
        result = build_sql_case_when([("amount", "greater_than", 100)], '"value"')
        assert result == """CASE WHEN "amount" > 100 THEN "value" END"""

    def test_equal_none_produces_is_null(self) -> None:
        result = build_sql_case_when([("col", "equal", None)], '"src"')
        assert "IS NULL" in result
        assert "= NULL" not in result
