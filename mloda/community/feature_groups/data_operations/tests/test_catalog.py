"""Runtime catalog of built-in data operations (issue #247).

The ``data_operations`` package init must export a ``DataOperationsCatalog``
(implemented in a sibling ``catalog.py``) that describes every built-in data
operation and its per-framework capability. Capability is derived from the
mloda core match-time machinery: per concrete production class, a subtype is
supported on a framework iff ``match_feature_group_criteria`` is truthy AND
``supports_compute_framework`` returns True for that framework. Frameworks
without an implementation are simply absent from ``OperationInfo.frameworks``.
"""

from __future__ import annotations

import dataclasses

import pytest

from mloda.community.feature_groups.data_operations import DataOperationsCatalog, OperationInfo


EXPECTED_OPERATION_NAMES: list[str] = [
    "aggregation",
    "binning",
    "datetime",
    "ema",
    "ffill",
    "frame_aggregate",
    "offset",
    "percentile",
    "point_arithmetic",
    "rank",
    "resample",
    "scalar_aggregate",
    "scalar_arithmetic",
    "sessionization",
    "string",
    "time_bucketization",
    "window_aggregation",
]

AGGREGATION_SUBTYPES: frozenset[str] = frozenset(
    {
        "sum",
        "avg",
        "mean",
        "count",
        "min",
        "max",
        "std",
        "var",
        "std_pop",
        "std_samp",
        "var_pop",
        "var_samp",
        "median",
        "mode",
        "nunique",
        "first",
        "last",
    }
)

FRAME_AGGREGATE_SUBTYPES: frozenset[str] = frozenset(
    {
        "rolling",
        "time:second",
        "time:minute",
        "time:hour",
        "time:day",
        "time:week",
        "time:month",
        "time:year",
        "cumulative",
        "expanding",
    }
)

OFFSET_SUBTYPES: frozenset[str] = frozenset({"lag", "lead", "diff", "pct_change", "first_value", "last_value"})


# ---------------------------------------------------------------------------
# OperationInfo shape
# ---------------------------------------------------------------------------


class TestOperationInfoShape:
    def test_operation_info_is_dataclass(self) -> None:
        """OperationInfo is a dataclass type."""
        assert dataclasses.is_dataclass(OperationInfo)

    def test_operation_info_field_names(self) -> None:
        """OperationInfo exposes exactly the documented fields."""
        field_names = {f.name for f in dataclasses.fields(OperationInfo)}
        assert field_names == {"name", "prefix_pattern", "subtype_label", "subtypes", "frameworks"}

    def test_operation_info_is_frozen(self) -> None:
        """OperationInfo instances are immutable (frozen dataclass)."""
        info = DataOperationsCatalog.get("aggregation")
        with pytest.raises(dataclasses.FrozenInstanceError):
            info.name = "renamed"  # type: ignore[misc]

    def test_field_value_types(self) -> None:
        """Fields carry the documented runtime types."""
        info = DataOperationsCatalog.get("aggregation")
        assert isinstance(info.name, str)
        assert isinstance(info.prefix_pattern, str)
        assert isinstance(info.subtype_label, str)
        assert isinstance(info.subtypes, tuple)
        assert isinstance(info.frameworks, dict)
        assert isinstance(info.frameworks["SqliteFramework"], frozenset)


# ---------------------------------------------------------------------------
# DataOperationsCatalog.list()
# ---------------------------------------------------------------------------


class TestList:
    def test_lists_all_operations_sorted_by_name(self) -> None:
        """list() returns all 17 built-in operations sorted by name."""
        names = [info.name for info in DataOperationsCatalog.list()]
        assert names == EXPECTED_OPERATION_NAMES

    def test_entries_are_operation_info(self) -> None:
        """Every list() entry is an OperationInfo."""
        assert all(isinstance(info, OperationInfo) for info in DataOperationsCatalog.list())

    def test_list_is_deterministic(self) -> None:
        """Two successive list() calls return equal results."""
        assert DataOperationsCatalog.list() == DataOperationsCatalog.list()


# ---------------------------------------------------------------------------
# DataOperationsCatalog.get()
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_returns_named_operation(self) -> None:
        """get(name) returns the OperationInfo for that operation."""
        info = DataOperationsCatalog.get("aggregation")
        assert isinstance(info, OperationInfo)
        assert info.name == "aggregation"

    def test_unknown_name_raises_value_error(self) -> None:
        """Unknown operation names raise ValueError."""
        with pytest.raises(ValueError):
            DataOperationsCatalog.get("no_such_operation")

    def test_unknown_name_error_lists_valid_operations(self) -> None:
        """The ValueError message echoes the rejected name and lists all valid operations."""
        with pytest.raises(ValueError) as exc_info:
            DataOperationsCatalog.get("no_such_operation")
        message = str(exc_info.value)
        assert "no_such_operation" in message
        for name in EXPECTED_OPERATION_NAMES:
            assert name in message


# ---------------------------------------------------------------------------
# Per-operation cells
# ---------------------------------------------------------------------------


class TestAggregationCell:
    def test_prefix_pattern(self) -> None:
        """aggregation carries the production base class PREFIX_PATTERN."""
        info = DataOperationsCatalog.get("aggregation")
        assert info.prefix_pattern == r".*__([\w]+)_agg$"

    def test_subtype_label(self) -> None:
        info = DataOperationsCatalog.get("aggregation")
        assert info.subtype_label == "agg type"

    def test_subtype_universe(self) -> None:
        """aggregation defines 17 subtypes including median, mean and mode."""
        info = DataOperationsCatalog.get("aggregation")
        assert info.subtypes is not None
        assert len(info.subtypes) == 17
        assert set(info.subtypes) == set(AGGREGATION_SUBTYPES)
        assert {"median", "mean", "mode"} <= set(info.subtypes)

    def test_sqlite_supports_only_native_functions(self) -> None:
        """SQLite supports exactly the six natively computable agg types."""
        info = DataOperationsCatalog.get("aggregation")
        assert info.frameworks["SqliteFramework"] == frozenset({"sum", "avg", "mean", "count", "min", "max"})

    def test_pandas_supports_full_universe(self) -> None:
        """Pandas supports the full 17-entry aggregation-type set."""
        pytest.importorskip("pandas")
        info = DataOperationsCatalog.get("aggregation")
        assert info.frameworks["PandasDataFrame"] == AGGREGATION_SUBTYPES

    def test_pyarrow_excludes_median_and_mode(self) -> None:
        """PyArrow supports everything except median and mode (15 entries)."""
        pytest.importorskip("pyarrow")
        info = DataOperationsCatalog.get("aggregation")
        pyarrow_set = info.frameworks["PyArrowTable"]
        assert pyarrow_set == AGGREGATION_SUBTYPES - {"median", "mode"}
        assert pyarrow_set is not None
        assert len(pyarrow_set) == 15


class TestStringCell:
    def test_sqlite_supports_only_trim_and_length(self) -> None:
        """SQLite string refuses upper/lower/reverse at match time; the catalog must reflect it."""
        info = DataOperationsCatalog.get("string")
        assert info.frameworks["SqliteFramework"] == frozenset({"trim", "length"})


class TestEmaCell:
    def test_has_no_subtype_axis(self) -> None:
        """ema has no subtype axis, so subtypes is None."""
        info = DataOperationsCatalog.get("ema")
        assert info.subtypes is None

    def test_framework_keys_and_values(self) -> None:
        """ema exists only on Pandas and Polars lazy; values are None (no subtype axis)."""
        pytest.importorskip("pandas")
        pytest.importorskip("polars")
        info = DataOperationsCatalog.get("ema")
        assert set(info.frameworks) == {"PandasDataFrame", "PolarsLazyDataFrame"}
        assert all(value is None for value in info.frameworks.values())


class TestPercentileCell:
    def test_unimplemented_frameworks_are_absent(self) -> None:
        """percentile has no SQLite or PyArrow implementation, so those keys are absent."""
        info = DataOperationsCatalog.get("percentile")
        assert "SqliteFramework" not in info.frameworks
        assert "PyArrowTable" not in info.frameworks

    def test_duckdb_is_present(self) -> None:
        """percentile is implemented on DuckDB."""
        pytest.importorskip("duckdb")
        info = DataOperationsCatalog.get("percentile")
        assert "DuckDBFramework" in info.frameworks


class TestResampleCell:
    def test_sqlite_is_absent(self) -> None:
        """resample has no SQLite implementation, so the key is absent."""
        info = DataOperationsCatalog.get("resample")
        assert "SqliteFramework" not in info.frameworks


class TestRankCell:
    def test_subtype_label(self) -> None:
        info = DataOperationsCatalog.get("rank")
        assert info.subtype_label == "rank type"

    def test_pandas_excludes_percent_rank(self) -> None:
        """Pandas percent_rank diverges from SQL semantics, so it is excluded."""
        pytest.importorskip("pandas")
        info = DataOperationsCatalog.get("rank")
        assert info.frameworks["PandasDataFrame"] == frozenset({"row_number", "rank", "dense_rank"})

    def test_duckdb_includes_percent_rank(self) -> None:
        """DuckDB supports SQL percent_rank natively."""
        pytest.importorskip("duckdb")
        info = DataOperationsCatalog.get("rank")
        duckdb_set = info.frameworks["DuckDBFramework"]
        assert duckdb_set is not None
        assert "percent_rank" in duckdb_set


class TestFrameAggregateCell:
    def test_subtype_label(self) -> None:
        info = DataOperationsCatalog.get("frame_aggregate")
        assert info.subtype_label == "frame type"

    def test_subtype_universe_uses_compound_time_subtypes(self) -> None:
        """frame_aggregate flattens (frame_type, time_unit) into 10 compound subtypes."""
        info = DataOperationsCatalog.get("frame_aggregate")
        assert info.subtypes is not None
        assert len(info.subtypes) == 10
        assert set(info.subtypes) == set(FRAME_AGGREGATE_SUBTYPES)

    def test_pandas_excludes_calendar_time_units(self) -> None:
        """Pandas supports rolling and day frames but not month or year time units."""
        pytest.importorskip("pandas")
        info = DataOperationsCatalog.get("frame_aggregate")
        pandas_set = info.frameworks["PandasDataFrame"]
        assert pandas_set is not None
        assert {"rolling", "time:day"} <= pandas_set
        assert "time:month" not in pandas_set
        assert "time:year" not in pandas_set

    def test_duckdb_includes_month_time_unit(self) -> None:
        """DuckDB supports calendar time units."""
        pytest.importorskip("duckdb")
        info = DataOperationsCatalog.get("frame_aggregate")
        duckdb_set = info.frameworks["DuckDBFramework"]
        assert duckdb_set is not None
        assert "time:month" in duckdb_set

    def test_pyarrow_is_absent(self) -> None:
        """frame_aggregate has no PyArrow implementation, so the key is absent."""
        info = DataOperationsCatalog.get("frame_aggregate")
        assert "PyArrowTable" not in info.frameworks


class TestOffsetCell:
    def test_subtype_label(self) -> None:
        info = DataOperationsCatalog.get("offset")
        assert info.subtype_label == "offset type"

    def test_subtype_universe(self) -> None:
        """offset defines exactly the six offset types."""
        info = DataOperationsCatalog.get("offset")
        assert info.subtypes is not None
        assert len(info.subtypes) == 6
        assert set(info.subtypes) == set(OFFSET_SUBTYPES)

    def test_pyarrow_is_absent(self) -> None:
        """offset has no PyArrow implementation, so the key is absent."""
        info = DataOperationsCatalog.get("offset")
        assert "PyArrowTable" not in info.frameworks


# ---------------------------------------------------------------------------
# DataOperationsCatalog.is_supported()
# ---------------------------------------------------------------------------


class TestIsSupported:
    def test_exact_cell_sqlite_median_false(self) -> None:
        """SQLite cannot compute a median aggregation."""
        assert DataOperationsCatalog.is_supported("aggregation", "median", "SqliteFramework") is False

    def test_exact_cell_duckdb_median_true(self) -> None:
        """DuckDB computes median natively."""
        pytest.importorskip("duckdb")
        assert DataOperationsCatalog.is_supported("aggregation", "median", "DuckDBFramework") is True

    def test_exact_cell_sqlite_mean_true(self) -> None:
        """SQLite supports mean via its AVG alias."""
        assert DataOperationsCatalog.is_supported("aggregation", "mean", "SqliteFramework") is True

    def test_operation_level_ema_pyarrow_false(self) -> None:
        """subtype=None asks whether the operation exists on the framework at all."""
        assert DataOperationsCatalog.is_supported("ema", framework="PyArrowTable") is False

    def test_operation_level_ema_pandas_true(self) -> None:
        pytest.importorskip("pandas")
        assert DataOperationsCatalog.is_supported("ema", framework="PandasDataFrame") is True

    def test_operation_level_percentile_sqlite_false(self) -> None:
        assert DataOperationsCatalog.is_supported("percentile", framework="SqliteFramework") is False

    def test_framework_matching_is_case_insensitive(self) -> None:
        """Framework names match case-insensitively."""
        assert DataOperationsCatalog.is_supported("aggregation", "sum", "sqliteframework") is True

    def test_any_framework_median_true(self) -> None:
        """framework=None asks whether at least one framework supports the subtype."""
        pytest.importorskip("duckdb")
        assert DataOperationsCatalog.is_supported("aggregation", subtype="median") is True

    def test_unknown_operation_raises_value_error_listing_operations(self) -> None:
        """Unknown operations raise ValueError listing the valid operation names."""
        with pytest.raises(ValueError) as exc_info:
            DataOperationsCatalog.is_supported("no_such_operation", "sum", "SqliteFramework")
        message = str(exc_info.value)
        assert "no_such_operation" in message
        assert "aggregation" in message
        assert "window_aggregation" in message

    def test_unknown_subtype_raises_value_error_listing_subtypes(self) -> None:
        """A subtype outside the operation's universe raises ValueError listing valid subtypes."""
        with pytest.raises(ValueError) as exc_info:
            DataOperationsCatalog.is_supported("aggregation", "bogus", "SqliteFramework")
        message = str(exc_info.value)
        assert "bogus" in message
        assert "median" in message
        assert "sum" in message

    def test_unknown_framework_returns_false(self) -> None:
        """Unknown or absent frameworks return False (open world, no error)."""
        assert DataOperationsCatalog.is_supported("aggregation", "sum", "NoSuchFramework") is False
