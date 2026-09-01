"""Exception hierarchy for the binary-model mixin and the exit-code-to-exception mapping
(contract: Errors). Every class is a ``ValueError`` subclass carrying ``code`` and ``message``.
"""

from __future__ import annotations

import json
from typing import ClassVar


class BinaryModelError(ValueError):
    """Base class for every binary-model error; ``CODE`` is the contract exit code, or ``None``
    for an error the mixin raises before any process runs."""

    CODE: ClassVar[int | None] = None

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
        self.code = type(self).CODE


class BinaryUnavailableError(BinaryModelError):
    """The binary could not be resolved or spawned at all; no contract exit code applies."""

    CODE: ClassVar[int | None] = None


class BinaryUsageError(BinaryModelError):
    """Contract exit code 1: bad flags or paths, malformed or unknown config keys."""

    CODE: ClassVar[int | None] = 1


class LicenseMissingError(BinaryModelError):
    """Contract exit code 2: no usable license source."""

    CODE: ClassVar[int | None] = 2


class LicenseInvalidError(BinaryModelError):
    """Contract exit code 3: license invalid, expired, or insufficient entitlement."""

    CODE: ClassVar[int | None] = 3


class UnsupportedError(BinaryModelError):
    """Contract exit code 4: unsupported operation or column type."""

    CODE: ClassVar[int | None] = 4


class DataError(BinaryModelError):
    """Contract exit code 5: malformed input or schema mismatch."""

    CODE: ClassVar[int | None] = 5


class BinaryInternalError(BinaryModelError):
    """Contract exit code 6, as reported by the binary itself (a crash, an unrecognized exit
    code, or an unparseable stderr error object)."""

    CODE: ClassVar[int | None] = 6


class BinaryTerminatedError(BinaryModelError):
    """Contract exit code 6, mixin-initiated: the binary was terminated (timeout or
    cancellation), not a binary-reported failure."""

    CODE: ClassVar[int | None] = 6


class OutputContractError(BinaryModelError):
    """Contract exit code 6, mixin-initiated: the binary's output violated the output contract
    (wrong schema, row count, or an unparseable stream) despite exit 0."""

    CODE: ClassVar[int | None] = 6


ERROR_CLASS_BY_CODE: dict[int, type[BinaryModelError]] = {
    1: BinaryUsageError,
    2: LicenseMissingError,
    3: LicenseInvalidError,
    4: UnsupportedError,
    5: DataError,
    6: BinaryInternalError,
}

_GENERIC_MESSAGE_FALLBACK = "binary reported code {code} without a usable message"


def _last_non_empty_line(stderr: bytes) -> str | None:
    text = stderr.decode("utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        if line.strip():
            return line
    return None


def error_from_exit(returncode: int, stderr: bytes) -> BinaryModelError:
    """Map a process exit code and its stderr to one ``BinaryModelError`` (contract: Errors).
    Never raises: any malformed input falls back to ``BinaryInternalError``."""
    line = _last_non_empty_line(stderr)
    if line is not None:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            code = payload.get("code")
            if (
                isinstance(code, int)
                and not isinstance(code, bool)
                and code == returncode
                and code in ERROR_CLASS_BY_CODE
            ):
                message = payload.get("message")
                if not isinstance(message, str) or not message:
                    message = _GENERIC_MESSAGE_FALLBACK.format(code=code)
                return ERROR_CLASS_BY_CODE[code](message)

    kind = "signal" if returncode < 0 else "exit code"
    return BinaryInternalError(f"binary failed with unrecognized {kind} {returncode} and no parseable error object")
