"""Single source for the binary-model contract's own constants and line/stream parsing
(contract: Invocation, Capabilities, Errors, Data handling), stdlib-only so the runtime mixin
never needs ``mloda.testing``; ``mloda.testing.binary_model`` re-exports these for the conformance
kit.
"""

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

# Bytes of stderr tail scanned for the error line, comfortably above the worst case (a `message`
# capped at MESSAGE_MAX_BYTES, grown up to about sixfold by `\u` escapes). A longer, out-of-contract
# error line falls back to BinaryInternalError.
_STDERR_TAIL_WINDOW_BYTES = 64 * 1024


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
    if lines and lines[-1] == "":
        lines.pop()
    return lines
