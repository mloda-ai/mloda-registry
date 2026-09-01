"""Binary-model conformance kit: contract-wide constants shared by the simulated CLI stub
(``mloda.testing.binary_model.simulated_binary``) and the conformance test surface
(``mloda.testing.binary_model.conformance``), so neither module keeps its own copy.

Kept intentionally small: only the values that are part of the binary-model interface contract
itself (its own version number, the fixed error-code table, the column-type vocabulary, and the
error-message size cap) live here. Everything else (the concrete "example_binary"/"hash" worked
example, the Arrow IPC wire mechanics, the "hash" reference algorithm, and the placeholder license
token shapes) lives in this package's other modules.
"""

from __future__ import annotations

# The binary-model interface contract's own version number, reported by --capabilities (contract:
# Capabilities).
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
