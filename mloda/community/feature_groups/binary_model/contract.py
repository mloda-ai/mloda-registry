"""Single source for the binary-model contract's constants and line/stream parsing, stdlib-only so
the runtime mixin never needs ``mloda.testing``; ``mloda.testing.binary_model`` re-exports these for
the conformance kit."""

from __future__ import annotations

# The contract's own version number, reported by --capabilities (contract: Capabilities).
CONTRACT_VERSION = 1

# The `--version` output's semver pattern, unanchored so callers can embed it in a larger
# expression (contract: Invocation).
VERSION_PATTERN = r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.+\-]+)?"

# The vocabulary's Arrow-type names (contract: Capabilities). ``utf8`` is pyarrow's 32-bit-offset
# string type, ``pa.string()`` -- not ``pa.large_string()`` or ``pa.string_view()``.
COLUMN_TYPES = frozenset({"int64", "float64", "utf8", "boolean"})

# Contract "Data handling": an error object's `message` is at most this many UTF-8 bytes.
MESSAGE_MAX_BYTES = 1024

# Contract "Errors" table: the exit code each error kind reports.
USAGE_ERROR = 1
LICENSE_MISSING = 2
LICENSE_INVALID = 3
UNSUPPORTED = 4
DATA_ERROR = 5
INTERNAL_ERROR = 6

# Bytes of stderr tail scanned for the error line, comfortably above the worst case (a `message`
# capped at MESSAGE_MAX_BYTES, grown up to about sixfold by `\u` escapes). A longer, out-of-contract
# error line falls back to BinaryInternalError.
_STDERR_TAIL_WINDOW_BYTES = 64 * 1024


def truncate_message(message: str) -> str:
    """Sanitize and cap ``message`` at ``MESSAGE_MAX_BYTES`` UTF-8 bytes, cutting only on a
    character boundary (contract: Data handling)."""
    return message.encode("utf-8", errors="replace")[:MESSAGE_MAX_BYTES].decode("utf-8", errors="ignore")


def stderr_excerpt(stderr: bytes, max_lines: int) -> str | None:
    """The last ``max_lines`` non-blank lines of stderr's tail window, in original order, joined by
    ``"\\n"`` and truncated; ``None`` when there is no non-blank line."""
    tail = stderr[-_STDERR_TAIL_WINDOW_BYTES:]
    kept: list[str] = []
    for line in reversed(tail.split(b"\n")):
        text = line.decode("utf-8", errors="replace")
        if text.strip():
            kept.append(text)
            if len(kept) >= max_lines:
                break
    if not kept:
        return None
    return truncate_message("\n".join(reversed(kept)))


def last_non_empty_stderr_line(stderr: bytes) -> str | None:
    """The last non-blank line of stderr's trailing tail window, split on ``b"\\n"`` only, never
    ``str.splitlines()``, which also splits on U+2028/U+2029/U+0085 and would corrupt a message
    containing one of them."""
    tail = stderr[-_STDERR_TAIL_WINDOW_BYTES:]
    for line in reversed(tail.split(b"\n")):
        text = line.decode("utf-8", errors="replace")
        if text.strip():
            return text
    return None


def split_output_lines(text: str) -> list[str]:
    """Split ``text`` on ``"\\n"`` only, dropping one trailing empty element."""
    # "\n" only: str.splitlines() also splits on U+2028/U+2029/U+0085, legal raw inside JSON strings.
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines
