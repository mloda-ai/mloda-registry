"""Binary-model conformance kit: contract-wide constants shared by the simulated CLI stub
(``simulated_binary.py``) and the conformance suite (``conformance.py``), so neither keeps its own
copy.

Kept intentionally small: only values that are part of the contract itself (version number, error
codes, column-type vocabulary, message size cap). Everything else (the worked example, Arrow IPC
mechanics, the "hash" algorithm, license-token shapes) lives in this package's other modules.
"""

from __future__ import annotations

# The contract's own version number, reported by --capabilities (contract: Capabilities).
CONTRACT_VERSION = 1

# Contract "Errors" table.
USAGE_ERROR = 1
LICENSE_MISSING = 2
LICENSE_INVALID = 3
UNSUPPORTED = 4
DATA_ERROR = 5
INTERNAL_ERROR = 6

# The vocabulary's Arrow-type names (contract: Capabilities). ``utf8`` is pyarrow's 32-bit-offset
# string type, ``pa.string()`` -- not ``pa.large_string()`` or ``pa.string_view()``.
COLUMN_TYPES = frozenset({"int64", "float64", "utf8", "boolean"})

# Contract "Data handling": an error object's `message` is at most this many UTF-8 bytes.
MESSAGE_MAX_BYTES = 1024

# Continuation marker (0xFFFFFFFF) then a zero-length message: the IPC end-of-stream marker
# (contract: Data). pyarrow's own reader tolerates a stream missing it, so this is checked on the
# raw trailing bytes instead (contract: Conformance).
IPC_END_OF_STREAM_MARKER = b"\xff\xff\xff\xff\x00\x00\x00\x00"
