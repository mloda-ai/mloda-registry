"""Tests for ``mixin.py``: ``BinaryModelMixin``, the entry point a FeatureGroup mixes in to run an
external binary as a model over Arrow IPC. Exercises every up-front rejection (each one before any
subprocess ``run``), the projection/casting/batching of the outgoing data, the per-invocation
directory lifecycle, output-contract verification, the mapping of binary-reported errors, the
returned table's shape, several happy-path shapes against the well-behaved simulated binary
(``mloda.testing.binary_model``), license overrides, and logging (contract: Capabilities, Data,
Configuration, License, Data handling, Errors).
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pytest

from mloda.community.feature_groups.binary_model import binary, mixin
from mloda.community.feature_groups.binary_model.errors import (
    BinaryInternalError,
    BinaryTerminatedError,
    BinaryUnavailableError,
    BinaryUsageError,
    DataError,
    LicenseInvalidError,
    LicenseMissingError,
    OutputContractError,
    UnsupportedError,
)
from mloda.community.feature_groups.binary_model.mixin import BinaryModelMixin
from mloda.community.feature_groups.binary_model.transport import TEMP_PARENT_NAME
from mloda.testing.binary_model.hash_reference import compute_expected_hash_column, hash_multi_column_case
from mloda.testing.binary_model.license_vectors import license_token_text

STUB_CMD = [sys.executable, "-m", "mloda.testing.binary_model.simulated_binary"]
FAULTY_CMD = [sys.executable, "-m", "mloda.community.feature_groups.binary_model.tests.faulty_binary"]
RESTRICTED_COLUMNS_CMD = [
    sys.executable,
    "-m",
    "mloda.community.feature_groups.binary_model.tests.mixin_fixtures",
    "--variant",
    "restricted_columns",
]
ECHO_UTF8_CMD = [
    sys.executable,
    "-m",
    "mloda.community.feature_groups.binary_model.tests.mixin_fixtures",
    "--variant",
    "echo_utf8",
]
PLUGIN_ID = "example_binary"
FAULTY_PLUGIN_ID = "faulty_binary"


@pytest.fixture(autouse=True)
def _clear_capability_cache_before_each_test() -> None:
    binary.clear_capability_cache()


class StubModel(BinaryModelMixin):
    BINARY_PLUGIN_ID = PLUGIN_ID
    BINARY_COMMAND_OVERRIDE = STUB_CMD
    LICENSE_KEY_OVERRIDE = license_token_text("valid", [PLUGIN_ID])


class _NoLicenseStubModel(BinaryModelMixin):
    """Same target binary as ``StubModel``, but no license override, so a test can control the
    license purely through the environment."""

    BINARY_PLUGIN_ID = PLUGIN_ID
    BINARY_COMMAND_OVERRIDE = STUB_CMD


class RestrictedColumnsModel(BinaryModelMixin):
    BINARY_PLUGIN_ID = "restricted_binary"
    BINARY_COMMAND_OVERRIDE = RESTRICTED_COLUMNS_CMD


class EchoUtf8Model(BinaryModelMixin):
    BINARY_PLUGIN_ID = "echo_utf8_binary"
    BINARY_COMMAND_OVERRIDE = ECHO_UTF8_CMD


def _faulty_model(mode: str, **class_attrs: Any) -> type[BinaryModelMixin]:
    """Build a ``BinaryModelMixin`` subclass targeting ``faulty_binary.py`` in the given mode."""
    attrs: dict[str, Any] = {
        "BINARY_PLUGIN_ID": FAULTY_PLUGIN_ID,
        "BINARY_COMMAND_OVERRIDE": [*FAULTY_CMD, "--mode", mode],
    }
    attrs.update(class_attrs)
    return cast(type[BinaryModelMixin], type(f"_Faulty_{mode}", (BinaryModelMixin,), attrs))


def _mloda_binary_children_for_current_pid() -> list[Path]:
    """Children of ``<temp>/mloda-binary`` named for the current process id: empty once every
    invocation directory this process created has been cleaned up (other pids are ignored)."""
    parent = Path(tempfile.gettempdir()) / TEMP_PARENT_NAME
    if not parent.is_dir():
        return []
    pid_prefix = f"{os.getpid()}-"
    return [child for child in parent.iterdir() if child.name.startswith(pid_prefix)]


# -------------------------------------------------------------------------------------------
# 1. Binary unavailable (checked before every other rejection)
# -------------------------------------------------------------------------------------------


class TestBinaryUnavailable:
    def test_missing_wheel_import_raises_binary_unavailable(self) -> None:
        class _NoWheelModel(BinaryModelMixin):
            BINARY_PLUGIN_ID = "definitely_not_an_installed_binary_model_plugin"

        table = pa.table({"col_a": ["alpha"]})
        with pytest.raises(BinaryUnavailableError):
            _NoWheelModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})

    def test_missing_override_path_raises_binary_unavailable(self, tmp_path: Path) -> None:
        class _MissingPathModel(BinaryModelMixin):
            BINARY_PLUGIN_ID = "whatever"
            BINARY_COMMAND_OVERRIDE = str(tmp_path / "does-not-exist")

        table = pa.table({"col_a": ["alpha"]})
        with pytest.raises(BinaryUnavailableError):
            _MissingPathModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})

    def test_capability_shape_violation_raises_binary_unavailable(self) -> None:
        model = _faulty_model("bad_capabilities")
        table = pa.table({"col_a": ["alpha"]})
        with pytest.raises(BinaryUnavailableError):
            model.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})

    def test_contract_mismatch_raises_binary_unavailable(self) -> None:
        model = _faulty_model("contract_2")
        table = pa.table({"col_a": ["alpha"]})
        with pytest.raises(BinaryUnavailableError):
            model.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})

    def test_binary_unavailable_precedes_usage_error_checks(self, tmp_path: Path) -> None:
        """An unresolvable binary is reported before ``input_columns`` is ever validated, even
        though an empty list would independently be a usage error (contract: Errors, check
        order)."""

        class _MissingPathModel(BinaryModelMixin):
            BINARY_PLUGIN_ID = "whatever"
            BINARY_COMMAND_OVERRIDE = str(tmp_path / "does-not-exist")

        table = pa.table({"col_a": ["alpha"]})
        with pytest.raises(BinaryUnavailableError):
            _MissingPathModel.run_binary_model(table, [], "hash", {}, {"result": "col_a_hash"})


# -------------------------------------------------------------------------------------------
# 2. input_columns validation
# -------------------------------------------------------------------------------------------


class TestInputColumnsValidation:
    def test_empty_input_columns_raises_usage_error(self) -> None:
        table = pa.table({"col_a": ["alpha"]})
        with pytest.raises(BinaryUsageError):
            StubModel.run_binary_model(table, [], "hash", {}, {"result": "col_a_hash"})

    def test_duplicate_input_columns_raises_usage_error(self) -> None:
        table = pa.table({"col_a": ["alpha"]})
        with pytest.raises(BinaryUsageError):
            StubModel.run_binary_model(table, ["col_a", "col_a"], "hash", {}, {"result": "col_a_hash"})

    def test_input_column_absent_from_table_raises_usage_error(self) -> None:
        table = pa.table({"col_a": ["alpha"]})
        with pytest.raises(BinaryUsageError):
            StubModel.run_binary_model(table, ["not_a_real_column"], "hash", {}, {"result": "col_a_hash"})


# -------------------------------------------------------------------------------------------
# 3. output_columns written-name validation
# -------------------------------------------------------------------------------------------


class TestOutputColumnsValidation:
    def test_written_names_not_unique_raises_usage_error(self) -> None:
        table = pa.table({"col_a": ["alpha"]})
        with pytest.raises(BinaryUsageError):
            StubModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "dup", "extra_key": "dup"})

    def test_written_name_colliding_with_input_column_raises_usage_error(self) -> None:
        table = pa.table({"col_a": ["alpha"]})
        with pytest.raises(BinaryUsageError):
            StubModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a"})

    def test_written_name_colliding_with_non_input_table_column_raises_usage_error(self) -> None:
        """Written names must be distinct from every column of the table, not just the columns
        named in ``input_columns`` (contract: Configuration)."""
        table = pa.table({"col_a": ["alpha"], "col_b": [1]})
        with pytest.raises(BinaryUsageError):
            StubModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_b"})


# -------------------------------------------------------------------------------------------
# 4. operation capability check
# -------------------------------------------------------------------------------------------


class TestOperationCapability:
    def test_operation_not_in_capabilities_raises_unsupported(self) -> None:
        table = pa.table({"col_a": ["alpha"]})
        with pytest.raises(UnsupportedError):
            StubModel.run_binary_model(table, ["col_a"], "not_a_real_operation", {}, {"result": "col_a_hash"})


# -------------------------------------------------------------------------------------------
# 5. column-type vocabulary (contract vocabulary, and per-binary column_types)
# -------------------------------------------------------------------------------------------


class TestColumnTypeVocabulary:
    @pytest.mark.parametrize(
        "bad_type, sample_values",
        [
            pytest.param(pa.int32(), [1, 2], id="int32_not_in_vocabulary"),
            pytest.param(pa.timestamp("us"), [0, 1], id="timestamp_not_in_vocabulary"),
        ],
    )
    def test_input_column_type_outside_vocabulary_raises_unsupported(
        self, bad_type: pa.DataType, sample_values: list[Any]
    ) -> None:
        table = pa.table({"col_a": pa.array(sample_values, type=bad_type)})
        with pytest.raises(UnsupportedError):
            StubModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})

    def test_list_type_input_column_raises_unsupported(self) -> None:
        table = pa.table({"col_a": pa.array([[1, 2], [3]], type=pa.list_(pa.int64()))})
        with pytest.raises(UnsupportedError):
            StubModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})

    def test_dictionary_encoded_input_column_raises_unsupported(self) -> None:
        dict_array = pa.array(["x", "y", "x"]).dictionary_encode()
        table = pa.table({"col_a": dict_array})
        with pytest.raises(UnsupportedError):
            StubModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})

    def test_column_type_absent_from_this_binarys_capabilities_raises_unsupported(self) -> None:
        """``"boolean"`` is in the contract's full vocabulary, but ``RestrictedColumnsModel``'s
        own ``capabilities.column_types`` omits it (contract: Capabilities)."""
        table = pa.table({"flag": [True, False]})
        with pytest.raises(UnsupportedError):
            RestrictedColumnsModel.run_binary_model(table, ["flag"], "hash", {}, {"result": "flag_hash"})


# -------------------------------------------------------------------------------------------
# 6. Oversized string cell
# -------------------------------------------------------------------------------------------


class TestOversizedStringCell:
    def test_string_cell_at_or_above_two_gib_raises_data_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mixin, "max_string_length", lambda column: 2**31)
        table = pa.table({"col_a": ["alpha", "beta"]})
        with pytest.raises(DataError):
            StubModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})

    def test_string_cell_just_below_the_threshold_is_not_rejected_on_size_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(mixin, "max_string_length", lambda column: 2**31 - 2)
        table = pa.table({"col_a": ["alpha", "beta"]})
        result = StubModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})
        assert result.num_rows == 2


# -------------------------------------------------------------------------------------------
# 7. Projection and metadata stripping
# -------------------------------------------------------------------------------------------


class TestProjectionAndMetadata:
    def test_extra_table_column_is_not_sent_to_the_binary(self) -> None:
        """A binary rejects an extra, unrequested column with a data error (contract: Data); a
        successful run therefore proves the mixin projected the table to ``input_columns`` before
        sending (contract: Data -- "the mixin projects its frame to those columns before
        sending")."""
        rows = {"col_a": ["alpha", "beta"]}
        table = pa.table({"col_a": rows["col_a"], "col_b": [1, 2]})
        result = StubModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})
        expected = compute_expected_hash_column(rows, ["col_a"], None)
        assert result.column("col_a_hash").to_pylist() == expected

    def test_schema_and_field_metadata_is_stripped_and_the_result_carries_none(self) -> None:
        field = pa.field("col_a", pa.string(), metadata={b"field_meta": b"x"})
        schema = pa.schema([field], metadata={b"pandas": b"x"})
        rows = {"col_a": ["alpha", "beta", "gamma"]}
        table = pa.table(rows, schema=schema)
        result = StubModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})
        expected = compute_expected_hash_column(rows, ["col_a"], None)
        assert result.column("col_a_hash").to_pylist() == expected
        assert not (result.schema.metadata or {})
        assert not (result.schema.field("col_a_hash").metadata or {})


# -------------------------------------------------------------------------------------------
# 8. large_string / string_view input columns cast to utf8 before sending
# -------------------------------------------------------------------------------------------


class TestStringTypeCasting:
    def test_large_string_input_column_is_cast_to_utf8_before_sending(self) -> None:
        """The stub rejects ``large_string`` outright (code 4); success proves the mixin cast it
        to ``utf8`` first (contract: Capabilities)."""
        rows = {"col_a": ["alpha", "beta", "gamma"]}
        table = pa.table({"col_a": pa.array(rows["col_a"], type=pa.large_string())})
        result = StubModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})
        expected = compute_expected_hash_column(rows, ["col_a"], None)
        assert result.column("col_a_hash").to_pylist() == expected

    def test_string_view_input_column_is_cast_to_utf8_before_sending(self) -> None:
        rows = {"col_a": ["alpha", "beta", "gamma"]}
        table = pa.table({"col_a": pa.array(rows["col_a"], type=pa.string_view())})
        result = StubModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})
        expected = compute_expected_hash_column(rows, ["col_a"], None)
        assert result.column("col_a_hash").to_pylist() == expected


# -------------------------------------------------------------------------------------------
# 9. Batching
# -------------------------------------------------------------------------------------------


class _TinyBatchStubModel(StubModel):
    MAX_BATCH_BYTES = 64


class TestBatching:
    def test_tiny_max_batch_bytes_still_returns_every_row_correctly(self) -> None:
        rows = {"col_a": [f"value-{i}" for i in range(50)]}
        table = pa.table(rows)
        result = _TinyBatchStubModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})
        expected = compute_expected_hash_column(rows, ["col_a"], None)
        assert result.num_rows == 50
        assert result.column("col_a_hash").to_pylist() == expected


# -------------------------------------------------------------------------------------------
# 10. InvocationDirectory lifecycle
# -------------------------------------------------------------------------------------------


class TestInvocationDirectoryCleanup:
    def test_directory_is_gone_after_a_successful_run(self) -> None:
        table = pa.table({"col_a": ["alpha"]})
        StubModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})
        assert _mloda_binary_children_for_current_pid() == []

    def test_directory_is_gone_after_a_failed_run(self) -> None:
        model = _faulty_model("garbage_stderr")
        table = pa.table({"col_a": ["alpha"]})
        with pytest.raises(BinaryInternalError):
            model.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})
        assert _mloda_binary_children_for_current_pid() == []


# -------------------------------------------------------------------------------------------
# 11. Output verification on exit 0
# -------------------------------------------------------------------------------------------


class TestOutputVerification:
    def test_unparseable_stream_raises_output_contract_error(self) -> None:
        model = _faulty_model("garbage_output")
        table = pa.table({"col_a": ["alpha"]})
        with pytest.raises(OutputContractError):
            model.run_binary_model(table, ["col_a"], "hash", {}, {"result": "result_out"})

    def test_wrong_field_name_raises_output_contract_error(self) -> None:
        model = _faulty_model("wrong_schema")
        table = pa.table({"col_a": ["alpha"]})
        with pytest.raises(OutputContractError):
            model.run_binary_model(table, ["col_a"], "hash", {}, {"result": "result_out"})

    def test_output_type_outside_vocabulary_raises_output_contract_error(self) -> None:
        """``faulty_binary``'s ``wrong_type`` mode emits ``int32``, which is outside
        ``column_types`` entirely (contract: Data)."""
        model = _faulty_model("wrong_type")
        table = pa.table({"col_a": ["alpha"]})
        with pytest.raises(OutputContractError):
            model.run_binary_model(table, ["col_a"], "hash", {}, {"result": "result_out"})

    def test_wrong_row_count_raises_output_contract_error(self) -> None:
        model = _faulty_model("wrong_row_count")
        table = pa.table({"col_a": ["alpha", "beta"]})
        with pytest.raises(OutputContractError):
            model.run_binary_model(table, ["col_a"], "hash", {}, {"result": "result_out"})

    def test_missing_end_of_stream_marker_is_tolerated_or_reported_as_output_contract_error(self) -> None:
        """pyarrow's own reader may still parse a stream missing the end-of-stream marker: this
        asserts the mixin either succeeds with correct rows or raises ``OutputContractError``, per
        the brief -- documented rather than a strict single expectation."""
        model = _faulty_model("missing_eos")
        table = pa.table({"col_a": ["alpha"]})
        try:
            result = model.run_binary_model(table, ["col_a"], "hash", {}, {"result": "result_out"})
        except OutputContractError:
            return
        assert result.column("result_out").to_pylist() == [0]


# -------------------------------------------------------------------------------------------
# 12. Binary-reported errors mapped by class
# -------------------------------------------------------------------------------------------


class TestBinaryReportedErrors:
    def test_missing_license_raises_license_missing_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MLODA_LICENSE_FILE", raising=False)
        monkeypatch.delenv("MLODA_LICENSE_KEY", raising=False)
        table = pa.table({"col_a": ["alpha"]})
        with pytest.raises(LicenseMissingError):
            _NoLicenseStubModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})

    def test_expired_license_raises_license_invalid_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLODA_LICENSE_KEY", license_token_text("expired", [PLUGIN_ID]))
        table = pa.table({"col_a": ["alpha"]})
        with pytest.raises(LicenseInvalidError):
            _NoLicenseStubModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})

    def test_garbage_stderr_raises_binary_internal_error(self) -> None:
        model = _faulty_model("garbage_stderr")
        table = pa.table({"col_a": ["alpha"]})
        with pytest.raises(BinaryInternalError):
            model.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})

    def test_signal_terminated_binary_raises_binary_internal_error(self) -> None:
        model = _faulty_model("signal")
        table = pa.table({"col_a": ["alpha"]})
        with pytest.raises(BinaryInternalError):
            model.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})

    def test_hanging_binary_raises_binary_terminated_error(self) -> None:
        model = _faulty_model("hang", BINARY_TIMEOUT_SECONDS=0.5)
        table = pa.table({"col_a": ["alpha"]})
        started = time.monotonic()
        with pytest.raises(BinaryTerminatedError):
            model.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})
        elapsed = time.monotonic() - started
        assert elapsed < 5.0, f"expected termination well before the 60s hang, took {elapsed}s"


# -------------------------------------------------------------------------------------------
# 13. Return value shape
# -------------------------------------------------------------------------------------------


class TestReturnValueShape:
    def test_result_contains_only_output_columns_row_aligned(self) -> None:
        rows = {"col_a": ["alpha", "beta", "gamma"]}
        table = pa.table({"col_a": rows["col_a"], "col_b": [1, 2, 3]})
        result = StubModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})
        assert result.schema.names == ["col_a_hash"]
        assert result.num_rows == table.num_rows
        expected = compute_expected_hash_column(rows, ["col_a"], None)
        assert result.column("col_a_hash").to_pylist() == expected

    def test_input_table_is_not_mutated(self) -> None:
        table = pa.table({"col_a": ["alpha", "beta"]})
        original_schema = table.schema
        StubModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})
        assert table.schema.equals(original_schema)
        assert table.column("col_a").to_pylist() == ["alpha", "beta"]

    def test_int64_output_is_not_cast_even_when_the_table_has_a_large_string_column(self) -> None:
        table = pa.table({"col_a": pa.array(["alpha", "beta"], type=pa.large_string())})
        result = StubModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})
        assert result.schema.field("col_a_hash").type == pa.int64()

    def test_utf8_output_is_cast_to_large_string_when_the_table_has_a_large_string_column(self) -> None:
        table = pa.table({"col_a": pa.array(["alpha", "beta"], type=pa.large_string())})
        result = EchoUtf8Model.run_binary_model(table, ["col_a"], "echo", {}, {"echo": "col_a_echo"})
        assert result.schema.field("col_a_echo").type == pa.large_string()
        assert result.column("col_a_echo").to_pylist() == ["alpha", "beta"]

    def test_utf8_output_is_not_cast_when_the_table_has_no_large_string_column(self) -> None:
        table = pa.table({"col_a": ["alpha", "beta"]})  # plain pa.string()
        result = EchoUtf8Model.run_binary_model(table, ["col_a"], "echo", {}, {"echo": "col_a_echo"})
        assert result.schema.field("col_a_echo").type == pa.string()


# -------------------------------------------------------------------------------------------
# 14. Happy paths against the simulated binary
# -------------------------------------------------------------------------------------------


class _FileTransportStubModel(StubModel):
    FILE_TRANSPORT_THRESHOLD_BYTES = 0


class TestHappyPaths:
    def test_single_utf8_column(self) -> None:
        rows = {"col_a": ["alpha", "beta", "gamma"]}
        table = pa.table(rows)
        result = StubModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})
        expected = compute_expected_hash_column(rows, ["col_a"], None)
        assert result.column("col_a_hash").to_pylist() == expected

    def test_five_column_mixed_case_with_null(self) -> None:
        case = hash_multi_column_case(key=None, output_column_name="multi_hash", make_config=lambda **kwargs: kwargs)
        table = pa.Table.from_pydict(case["rows"], schema=case["schema"])
        result = StubModel.run_binary_model(table, case["input_columns"], "hash", {}, case["output_columns"])
        assert result.column("multi_hash").to_pylist() == case["expected"]

    def test_schema_only_zero_row_table(self) -> None:
        table = pa.table({"col_a": pa.array([], type=pa.string())})
        result = StubModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})
        assert result.num_rows == 0
        assert result.schema.names == ["col_a_hash"]
        assert result.schema.field("col_a_hash").type == pa.int64()

    def test_parameters_key_changes_the_result_per_the_reference_algorithm(self) -> None:
        rows = {"col_a": ["alpha", "beta"]}
        table = pa.table(rows)
        result = StubModel.run_binary_model(table, ["col_a"], "hash", {"key": "k"}, {"result": "col_a_hash"})
        expected = compute_expected_hash_column(rows, ["col_a"], "k")
        assert result.column("col_a_hash").to_pylist() == expected

    def test_file_transport_path_gives_identical_results(self) -> None:
        rows = {"col_a": ["alpha", "beta", "gamma"]}
        table = pa.table(rows)
        result = _FileTransportStubModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})
        expected = compute_expected_hash_column(rows, ["col_a"], None)
        assert result.column("col_a_hash").to_pylist() == expected


# -------------------------------------------------------------------------------------------
# 15. License overrides
# -------------------------------------------------------------------------------------------


class TestLicenseOverrides:
    def test_license_file_override_works_with_no_license_env_vars_set(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("MLODA_LICENSE_FILE", raising=False)
        monkeypatch.delenv("MLODA_LICENSE_KEY", raising=False)
        license_path = tmp_path / "license.txt"
        license_path.write_text(license_token_text("valid", [PLUGIN_ID]), encoding="utf-8")

        class _FileOverrideModel(BinaryModelMixin):
            BINARY_PLUGIN_ID = PLUGIN_ID
            BINARY_COMMAND_OVERRIDE = STUB_CMD
            LICENSE_FILE_OVERRIDE = str(license_path)

        table = pa.table({"col_a": ["alpha"]})
        result = _FileOverrideModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})
        assert result.num_rows == 1

    def test_license_key_override_works_with_no_license_env_vars_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MLODA_LICENSE_FILE", raising=False)
        monkeypatch.delenv("MLODA_LICENSE_KEY", raising=False)

        class _KeyOverrideModel(BinaryModelMixin):
            BINARY_PLUGIN_ID = PLUGIN_ID
            BINARY_COMMAND_OVERRIDE = STUB_CMD
            LICENSE_KEY_OVERRIDE = license_token_text("valid", [PLUGIN_ID])

        table = pa.table({"col_a": ["alpha"]})
        result = _KeyOverrideModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})
        assert result.num_rows == 1

    def test_license_override_beats_a_conflicting_expired_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLODA_LICENSE_KEY", license_token_text("expired", [PLUGIN_ID]))

        class _OverrideBeatsEnvModel(BinaryModelMixin):
            BINARY_PLUGIN_ID = PLUGIN_ID
            BINARY_COMMAND_OVERRIDE = STUB_CMD
            LICENSE_KEY_OVERRIDE = license_token_text("valid", [PLUGIN_ID])

        table = pa.table({"col_a": ["alpha"]})
        result = _OverrideBeatsEnvModel.run_binary_model(table, ["col_a"], "hash", {}, {"result": "col_a_hash"})
        assert result.num_rows == 1


# -------------------------------------------------------------------------------------------
# 16. Logging
# -------------------------------------------------------------------------------------------


class TestLogging:
    def test_run_logs_plugin_id_and_exit_code_never_the_operation_parameters(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A distinctive marker, not a real secret: named to avoid tripping bandit's B105
        # (hardcoded-password) heuristic, which keys off variable names like "secret"/"password".
        distinctive_parameter_value = "SECRET_KEY_x9"
        with caplog.at_level(logging.DEBUG, logger="mloda.community.feature_groups.binary_model"):
            table = pa.table({"col_a": ["alpha"]})
            StubModel.run_binary_model(
                table, ["col_a"], "hash", {"key": distinctive_parameter_value}, {"result": "col_a_hash"}
            )
        messages = [record.getMessage() for record in caplog.records]
        assert any(PLUGIN_ID in message for message in messages), f"plugin_id not logged: {messages!r}"
        assert any("0" in message for message in messages), f"exit code not logged: {messages!r}"
        assert not any(distinctive_parameter_value in message for message in messages), (
            f"secret leaked into logs: {messages!r}"
        )
