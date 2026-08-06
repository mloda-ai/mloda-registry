"""Unit tests for mask_utils shared utilities."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.community.feature_groups.data_operations.mask_utils import (
    build_polars_mask_expr,
    build_sql_case_when,
    parse_mask_spec,
)


class TestParseMaskSpec:
    def test_none_returns_none(self) -> None:
        assert parse_mask_spec(None) is None

    @pytest.mark.parametrize(
        ("mask_option", "expected"),
        [
            pytest.param(("col", "equal", "X"), [("col", "equal", "X")], id="single_tuple"),
            pytest.param(
                [("a", "equal", 1), ("b", "greater_equal", 10)],
                [("a", "equal", 1), ("b", "greater_equal", 10)],
                id="list_of_tuples",
            ),
            pytest.param(("col", "is_in", ["A", "B"]), [("col", "is_in", ["A", "B"])], id="is_in_list"),
            pytest.param(("col", "greater_than", 10), [("col", "greater_than", 10)], id="greater_than"),
            # A 2-element tuple sets val=None, which is valid for 'equal'.
            pytest.param(("col", "equal"), [("col", "equal", None)], id="two_element_equal_none"),
        ],
    )
    def test_accepts_valid_spec(self, mask_option: Any, expected: list[tuple[str, str, Any]]) -> None:
        result = parse_mask_spec(mask_option)
        assert result == expected

    @pytest.mark.parametrize(
        ("mask_option", "message"),
        [
            pytest.param(("col", "not_equal", "X"), "Unsupported mask operator", id="invalid_operator"),
            pytest.param("bad", "must be a tuple or list", id="invalid_type"),
            pytest.param(("a",), "2 or 3 elements", id="wrong_tuple_length"),
            pytest.param((123, "equal", "X"), "column must be a string", id="non_string_column"),
            pytest.param(("col", "is_in", "DE"), "is_in values must be a list", id="is_in_string"),
            pytest.param(("col", "is_in", b"DE"), "is_in values must be a list", id="is_in_bytes"),
            pytest.param(("col", "is_in", []), "must not be empty", id="is_in_empty_list"),
            pytest.param(("col", "is_in", set()), "must not be empty", id="is_in_empty_set"),
            # A 2-element tuple is only valid for 'equal', not for other operators.
            pytest.param(("col", "greater_than"), "only valid for 'equal'", id="two_element_greater_than"),
            pytest.param(("col", "is_in"), "only valid for 'equal'", id="two_element_is_in"),
        ],
    )
    def test_rejects_invalid_spec(self, mask_option: Any, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            parse_mask_spec(mask_option)

    def test_is_in_set_accepted(self) -> None:
        result = parse_mask_spec(("col", "is_in", {"A", "B"}))
        assert result is not None
        assert result[0][1] == "is_in"

    def test_unsupported_value_type_rejected(self) -> None:
        from datetime import datetime

        with pytest.raises(ValueError, match="Mask value must be"):
            parse_mask_spec(("col", "equal", datetime.now()))


class TestBuildPolarsMaskExpr:
    def test_single_equal(self) -> None:
        pl = pytest.importorskip("polars")

        expr = build_polars_mask_expr([("status", "equal", "active")])
        df = pl.DataFrame({"status": ["active", "inactive", "active"]})
        result = df.lazy().filter(expr).collect()
        assert result.shape == (2, 1)
        assert result["status"].to_list() == ["active", "active"]

    def test_multiple_conditions(self) -> None:
        pl = pytest.importorskip("polars")

        expr = build_polars_mask_expr([("cat", "equal", "X"), ("val", "greater_equal", 10)])
        df = pl.DataFrame({"cat": ["X", "X", "Y"], "val": [15, 5, 20]})
        result = df.lazy().filter(expr).collect()
        assert result.shape == (1, 2)

    def test_is_in(self) -> None:
        pl = pytest.importorskip("polars")

        expr = build_polars_mask_expr([("col", "is_in", ["a", "b"])])
        df = pl.DataFrame({"col": ["a", "c", "b"]})
        result = df.lazy().filter(expr).collect()
        assert result.shape == (2, 1)

    def test_all_comparison_operators(self) -> None:
        pl = pytest.importorskip("polars")

        for op, test_val, expected_count in [
            ("greater_than", 2, 1),
            ("greater_equal", 2, 2),
            ("less_than", 2, 1),
            ("less_equal", 2, 2),
        ]:
            expr = build_polars_mask_expr([("x", op, test_val)])
            df = pl.DataFrame({"x": [1, 2, 3]})
            result = df.lazy().filter(expr).collect()
            assert result.shape[0] == expected_count, f"Failed for {op}"


class TestBuildSqlCaseWhen:
    @pytest.mark.parametrize(
        ("column", "operator", "value", "expected"),
        [
            pytest.param(
                "status", "equal", "active", """CASE WHEN "status" = 'active' THEN "value" END""", id="single_equal"
            ),
            pytest.param("amount", "greater_than", 100, 'CASE WHEN "amount" > 100 THEN "value" END', id="greater_than"),
            pytest.param("amount", "less_than", 100, 'CASE WHEN "amount" < 100 THEN "value" END', id="less_than"),
            pytest.param("amount", "less_equal", 100, 'CASE WHEN "amount" <= 100 THEN "value" END', id="less_equal"),
            pytest.param(
                "amount", "greater_equal", 100, 'CASE WHEN "amount" >= 100 THEN "value" END', id="greater_equal"
            ),
        ],
    )
    def test_single_condition(self, column: str, operator: str, value: str | int, expected: str) -> None:
        result = build_sql_case_when([(column, operator, value)], '"value"')
        assert result == expected

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

    def test_equal_none_produces_is_null(self) -> None:
        result = build_sql_case_when([("col", "equal", None)], '"src"')
        assert "IS NULL" in result
        assert "= NULL" not in result
