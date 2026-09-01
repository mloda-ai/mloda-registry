"""Tests for the binary-model mixin's exception hierarchy and its exit-code-to-exception mapping
(contract: Errors). ``error_from_exit`` never raises: every malformed-input case it must tolerate
(unparseable stderr, an empty stderr, a code/returncode mismatch, a negative returncode, a JSON
array instead of an object) falls back to ``BinaryInternalError``.
"""

from __future__ import annotations

import json

import pytest

from mloda.community.feature_groups.binary_model.errors import (
    ERROR_CLASS_BY_CODE,
    BinaryInternalError,
    BinaryModelError,
    BinaryTerminatedError,
    BinaryUnavailableError,
    BinaryUsageError,
    DataError,
    LicenseInvalidError,
    LicenseMissingError,
    OutputContractError,
    UnsupportedError,
    error_from_exit,
)

# (error class, expected CODE): every subclass, including the two mixin-initiated ones
# (``BinaryTerminatedError``, ``OutputContractError``) that share code 6 with
# ``BinaryInternalError`` but are never returned by ``error_from_exit`` (contract: Errors).
_ALL_ERROR_CLASSES = [
    pytest.param(BinaryUnavailableError, None, id="BinaryUnavailableError"),
    pytest.param(BinaryUsageError, 1, id="BinaryUsageError"),
    pytest.param(LicenseMissingError, 2, id="LicenseMissingError"),
    pytest.param(LicenseInvalidError, 3, id="LicenseInvalidError"),
    pytest.param(UnsupportedError, 4, id="UnsupportedError"),
    pytest.param(DataError, 5, id="DataError"),
    pytest.param(BinaryInternalError, 6, id="BinaryInternalError"),
    pytest.param(BinaryTerminatedError, 6, id="BinaryTerminatedError"),
    pytest.param(OutputContractError, 6, id="OutputContractError"),
]


@pytest.mark.parametrize("error_class,expected_code", _ALL_ERROR_CLASSES)
def test_error_class_code_and_message(error_class: type[BinaryModelError], expected_code: int | None) -> None:
    exc = error_class("something went wrong")
    assert error_class.CODE == expected_code
    assert exc.code == expected_code
    assert exc.message == "something went wrong"
    assert str(exc) == "something went wrong"
    assert isinstance(exc, ValueError)
    assert isinstance(exc, BinaryModelError)


def test_base_class_code_is_none() -> None:
    assert BinaryModelError.CODE is None


def test_error_class_by_code_maps_binary_reported_codes_only() -> None:
    assert ERROR_CLASS_BY_CODE == {
        1: BinaryUsageError,
        2: LicenseMissingError,
        3: LicenseInvalidError,
        4: UnsupportedError,
        5: DataError,
        6: BinaryInternalError,
    }


def _stderr_line(code: int, message: str) -> bytes:
    return (json.dumps({"code": code, "message": message}) + "\n").encode("utf-8")


@pytest.mark.parametrize("code", [1, 2, 3, 4, 5, 6])
def test_error_from_exit_matches_reported_code(code: int) -> None:
    exc = error_from_exit(code, _stderr_line(code, "boom"))
    assert isinstance(exc, ERROR_CLASS_BY_CODE[code])
    assert exc.code == code
    assert exc.message == "boom"


def test_error_from_exit_ignores_earlier_diagnostic_lines() -> None:
    stderr = b"free-form diagnostic one\nanother diagnostic line\n" + _stderr_line(5, "malformed input")
    exc = error_from_exit(5, stderr)
    assert isinstance(exc, DataError)
    assert exc.message == "malformed input"


def test_error_from_exit_unparseable_stderr_becomes_internal_error() -> None:
    exc = error_from_exit(5, b"not json at all")
    assert isinstance(exc, BinaryInternalError)
    assert exc.code == 6
    assert "5" in exc.message


def test_error_from_exit_empty_stderr_becomes_internal_error() -> None:
    exc = error_from_exit(3, b"")
    assert isinstance(exc, BinaryInternalError)
    assert exc.code == 6
    assert "3" in exc.message


def test_error_from_exit_code_mismatch_becomes_internal_error() -> None:
    exc = error_from_exit(5, _stderr_line(3, "license expired"))
    assert isinstance(exc, BinaryInternalError)
    assert exc.code == 6
    assert "5" in exc.message


def test_error_from_exit_negative_returncode_becomes_internal_error() -> None:
    exc = error_from_exit(-9, b"")
    assert isinstance(exc, BinaryInternalError)
    assert exc.code == 6
    assert "-9" in exc.message


def test_error_from_exit_json_array_becomes_internal_error() -> None:
    exc = error_from_exit(5, b"[1, 2, 3]\n")
    assert isinstance(exc, BinaryInternalError)
    assert exc.code == 6
    assert "5" in exc.message


def test_error_from_exit_code_not_in_table_becomes_internal_error() -> None:
    exc = error_from_exit(7, _stderr_line(7, "whatever"))
    assert isinstance(exc, BinaryInternalError)
    assert exc.code == 6
    assert "7" in exc.message


def test_error_from_exit_never_raises_on_non_utf8_stderr() -> None:
    exc = error_from_exit(5, b"\xff\xfe not valid utf-8 \x00")
    assert isinstance(exc, BinaryInternalError)
    assert exc.code == 6
