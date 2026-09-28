"""Tests for the binary-model mixin's exception hierarchy and its exit-code-to-exception mapping
(contract: Errors). ``error_from_exit`` never raises: every malformed-input case it must tolerate
(unparseable stderr, an empty stderr, a code/returncode mismatch, a negative returncode, a JSON
array instead of an object) falls back to ``BinaryInternalError``.
"""

from __future__ import annotations

import json

import pytest

from mloda.community.feature_groups.binary_model import contract, errors
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


@pytest.mark.parametrize(
    "code_name, error_class",
    [
        ("USAGE_ERROR", BinaryUsageError),
        ("LICENSE_MISSING", LicenseMissingError),
        ("LICENSE_INVALID", LicenseInvalidError),
        ("UNSUPPORTED", UnsupportedError),
        ("DATA_ERROR", DataError),
        ("INTERNAL_ERROR", BinaryInternalError),
    ],
)
def test_error_class_by_code_matches_contract_exit_code_constants(
    code_name: str, error_class: type[BinaryModelError]
) -> None:
    """Each contract exit-code constant (`contract.USAGE_ERROR`, ..., `contract.INTERNAL_ERROR`)
    must map to its error class in `ERROR_CLASS_BY_CODE`, the single source for both the mixin and
    the testing kit (contract: Errors). ``getattr`` is looked up at test-run time so a missing
    constant fails this one case, not the whole module's collection."""
    code = getattr(contract, code_name)
    assert ERROR_CLASS_BY_CODE[code] is error_class


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


def test_error_from_exit_never_raises_on_a_lone_surrogate_message() -> None:
    """A JSON ``"\\ud800"`` escape decodes to a lone surrogate, which cannot be encoded as UTF-8. The
    mapping must still return the class for the matching code with a sanitized, encodable message."""
    exc = error_from_exit(5, _stderr_line(5, "\ud800"))
    assert isinstance(exc, DataError)
    assert exc.code == 5
    assert exc.message == "?"


def test_error_from_exit_never_raises_on_deeply_nested_json() -> None:
    """Deeply nested JSON makes ``json.loads`` raise ``RecursionError`` (not a ``ValueError``) on CPython
    3.10 through 3.13. The mapping must still return a ``BinaryInternalError``; the exception type that
    ``json.loads`` raises is deliberately not asserted."""
    exc = error_from_exit(5, b"[" * 20000 + b"]" * 20000 + b"\n")
    assert isinstance(exc, BinaryInternalError)
    assert exc.code == 6


def test_error_from_exit_never_raises_on_an_oversized_integer_code() -> None:
    """A 5000-digit ``code`` exceeds the interpreter's int string conversion limit on some Python
    versions (``json.loads`` raises ``ValueError``); on others it parses. Either way the result is a
    ``BinaryInternalError`` and nothing is raised."""
    exc = error_from_exit(5, b'{"code":' + b"9" * 5000 + b"}")
    assert isinstance(exc, BinaryInternalError)
    assert exc.code == 6


@pytest.mark.parametrize(
    "ch",
    ["\u2028", "\u2029", "\u0085"],
    ids=["line_separator", "paragraph_separator", "next_line"],
)
def test_error_from_exit_message_containing_unicode_line_boundary_is_not_split(ch: str) -> None:
    """``str.splitlines()`` treats U+2028 (LINE SEPARATOR), U+2029 (PARAGRAPH SEPARATOR) and
    U+0085 (NEXT LINE) as line boundaries, unlike a real stderr line terminator, which is always
    ``\\n``. A message containing one of these characters must still map to its reported class
    with the character intact, not be corrupted by splitting on it (contract: Errors, Data
    handling)."""
    message = f"bad column a{ch}b"
    stderr = json.dumps({"code": 5, "message": message}, ensure_ascii=False).encode("utf-8") + b"\n"
    exc = error_from_exit(5, stderr)
    assert isinstance(exc, DataError)
    assert ch in exc.message


def test_error_from_exit_matches_reported_code_after_many_kib_of_diagnostics() -> None:
    """The error line still maps to its reported class even after more than 64 KiB of earlier
    free-form diagnostic lines (contract: Errors)."""
    stderr = (b"free-form diagnostic line\n" * 4000) + _stderr_line(5, "malformed input")
    assert len(stderr) > 64 * 1024
    exc = error_from_exit(5, stderr)
    assert isinstance(exc, DataError)
    assert exc.message == "malformed input"


def test_error_from_exit_matches_reported_code_with_crlf_line_ending() -> None:
    """A CRLF-terminated error line still maps to its reported class (contract: Errors)."""
    stderr = json.dumps({"code": 5, "message": "malformed input"}).encode("utf-8") + b"\r\n"
    exc = error_from_exit(5, stderr)
    assert isinstance(exc, DataError)
    assert exc.message == "malformed input"


def test_error_from_exit_truncates_a_long_message_on_a_utf8_character_boundary() -> None:
    """A ``message`` longer than 1024 UTF-8 bytes is truncated to at most 1024 bytes, cutting only
    on a character boundary (contract: Data handling). The snowman character is 3 bytes in UTF-8,
    so a naive ``bytes[:1024]`` slice lands mid-character and fails to decode -- proving the cut
    respected UTF-8 boundaries, not raw byte count."""
    long_message = "☃" * 500  # 1500 bytes in UTF-8
    with pytest.raises(UnicodeDecodeError):
        long_message.encode("utf-8")[:1024].decode("utf-8")

    exc = error_from_exit(5, _stderr_line(5, long_message))

    encoded = exc.message.encode("utf-8")
    assert len(encoded) <= 1024
    assert len(exc.message) > 0
    assert exc.message != long_message


def test_last_non_empty_stderr_line_skips_trailing_blank_lines() -> None:
    """``contract.last_non_empty_stderr_line`` skips over trailing blank lines to find the last
    non-blank one (contract: Errors, Data handling). The U+2028-kept and empty-stderr cases are
    already covered indirectly via ``error_from_exit`` elsewhere in this file."""
    assert contract.last_non_empty_stderr_line(b"one\ntwo\n\n\n") == "two"


def test_last_non_empty_stderr_line_only_scans_the_trailing_64_kib_tail() -> None:
    """A final line longer than the 64 KiB tail window is cut to exactly that window's length, a
    single ASCII line with no embedded newline (contract: Data handling)."""
    long_line = b"x" * (70 * 1024)
    result = contract.last_non_empty_stderr_line(long_line)
    assert result is not None
    assert len(result) == 64 * 1024


# -- `errors.reported_error`: a pure, non-raising mapping used by the testing kit's own assertions --


def test_reported_error_valid_error_object_returns_mapped_error_instance() -> None:
    """A stderr line whose code matches the returncode and lies within 1..6 maps to the same class
    as ``error_from_exit`` (contract: Errors)."""
    result = errors.reported_error(5, _stderr_line(5, "malformed input"))
    assert isinstance(result, DataError)
    assert result.message == "malformed input"


@pytest.mark.parametrize(
    "returncode, stderr",
    [
        pytest.param(1, b'{"code": 5.0, "message": "m"}\n', id="float_code"),
        pytest.param(1, b'{"code": true, "message": "m"}\n', id="boolean_code"),
        pytest.param(5, b'{"code": 3, "message": "m"}\n', id="code_mismatch"),
        pytest.param(5, b"not json at all\n", id="non_json"),
        pytest.param(5, b"", id="empty_stderr"),
    ],
)
def test_reported_error_malformed_or_mismatched_returns_none(returncode: int, stderr: bytes) -> None:
    """A code that isn't a plain int (a float, a bool), doesn't match the returncode, or isn't
    parseable JSON at all must map to ``None``, never raise (contract: Errors)."""
    assert errors.reported_error(returncode, stderr) is None
