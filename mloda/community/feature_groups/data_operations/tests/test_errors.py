"""Tests for shared error-construction helpers used by data operations."""

from __future__ import annotations

import pytest

from mloda.community.feature_groups.data_operations.errors import (
    unsupported_agg_type_error,
    unsupported_frame_type_error,
)


class TestUnsupportedAggTypeError:
    def test_returns_value_error(self) -> None:
        """The helper returns a ValueError instance (not raises it)."""
        err = unsupported_agg_type_error("foo", ["sum", "avg"])
        assert isinstance(err, ValueError)

    def test_message_quotes_rejected_value(self) -> None:
        """Rejected value is quoted via repr() so whitespace/empty is visible."""
        err = unsupported_agg_type_error("foo", ["sum"])
        assert "'foo'" in str(err)

    def test_message_lists_supported_sorted(self) -> None:
        """Supported types appear alphabetically sorted in the message."""
        err = unsupported_agg_type_error("foo", ["sum", "avg", "count"])
        assert "Supported types: avg, count, sum." in str(err)

    def test_message_deduplicates_supported(self) -> None:
        """Duplicate entries in supported are deduplicated."""
        err = unsupported_agg_type_error("foo", ["sum", "sum", "avg"])
        assert "Supported types: avg, sum." in str(err)

    def test_message_includes_framework_when_given(self) -> None:
        err = unsupported_agg_type_error("foo", ["sum"], framework="DuckDB")
        assert "Unsupported aggregation type for DuckDB:" in str(err)

    def test_message_includes_operation_when_given(self) -> None:
        err = unsupported_agg_type_error(
            "foo",
            ["sum"],
            framework="SQLite",
            operation="frame aggregate",
        )
        assert "Unsupported aggregation type for SQLite frame aggregate:" in str(err)

    def test_message_without_framework_or_operation(self) -> None:
        err = unsupported_agg_type_error("foo", ["sum"])
        assert str(err) == "Unsupported aggregation type: 'foo'. Supported types: sum."

    def test_accepts_dict_keys_view(self) -> None:
        """Callers pass ``dict.keys()`` directly: ensure it works."""
        supported = {"sum": "SUM", "avg": "AVG"}
        err = unsupported_agg_type_error("foo", supported.keys(), framework="DuckDB")
        assert "Supported types: avg, sum." in str(err)

    def test_accepts_set(self) -> None:
        """Callers pass pre-built sets: ensure iteration order is normalised."""
        err = unsupported_agg_type_error("foo", {"sum", "avg", "count"})
        # Order must be alphabetical regardless of set insertion order.
        assert "Supported types: avg, count, sum." in str(err)

    def test_empty_string_agg_type_is_visible(self) -> None:
        """Empty agg_type must show up as '' (not a silent empty substring)."""
        err = unsupported_agg_type_error("", ["sum"])
        assert "''" in str(err)

    def test_raisable(self) -> None:
        """The returned object must be usable as a raise target."""
        with pytest.raises(ValueError, match="Supported types: avg, sum"):
            raise unsupported_agg_type_error("foo", ["sum", "avg"])


class TestUnsupportedFrameTypeError:
    def test_returns_value_error(self) -> None:
        err = unsupported_frame_type_error("sliding", ["rolling"])
        assert isinstance(err, ValueError)

    def test_message_quotes_rejected_value(self) -> None:
        err = unsupported_frame_type_error("sliding", ["rolling"])
        assert "'sliding'" in str(err)

    def test_message_lists_supported_sorted(self) -> None:
        err = unsupported_frame_type_error(
            "sliding",
            {"rolling", "cumulative", "expanding"},
        )
        assert "Supported types: cumulative, expanding, rolling." in str(err)

    def test_message_includes_framework_when_given(self) -> None:
        err = unsupported_frame_type_error("sliding", ["rolling"], framework="DuckDB")
        assert "Unsupported frame type for DuckDB:" in str(err)

    def test_raisable(self) -> None:
        with pytest.raises(ValueError, match="Unsupported frame type for Pandas"):
            raise unsupported_frame_type_error(
                "sliding",
                ["rolling"],
                framework="Pandas",
            )
