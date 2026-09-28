"""Binary-model conformance kit: contract-wide constants shared by the simulated CLI stub
(``simulated_binary.py``) and the conformance suite (``conformance.py``), so neither keeps its own
copy.

``CONTRACT_VERSION``, ``VERSION_PATTERN``, ``COLUMN_TYPES`` and ``MESSAGE_MAX_BYTES`` are re-exported
from the mixin's own ``mloda.community.feature_groups.binary_model.contract``, the single source of
the contract rules; the kit depends on the mixin, never the reverse. Everything else (error codes,
the worked example, Arrow IPC mechanics, the "hash" algorithm, license-token shapes) lives in this
package's own modules.
"""

from __future__ import annotations

from mloda.community.feature_groups.binary_model.contract import COLUMN_TYPES as COLUMN_TYPES
from mloda.community.feature_groups.binary_model.contract import CONTRACT_VERSION as CONTRACT_VERSION
from mloda.community.feature_groups.binary_model.contract import MESSAGE_MAX_BYTES as MESSAGE_MAX_BYTES
from mloda.community.feature_groups.binary_model.contract import VERSION_PATTERN as VERSION_PATTERN

# Contract "Errors" table.
USAGE_ERROR = 1
LICENSE_MISSING = 2
LICENSE_INVALID = 3
UNSUPPORTED = 4
DATA_ERROR = 5
INTERNAL_ERROR = 6

# Continuation marker (0xFFFFFFFF) then a zero-length message: the IPC end-of-stream marker
# (contract: Data). pyarrow's own reader tolerates a stream missing it, so this is checked on the
# raw trailing bytes instead (contract: Conformance).
IPC_END_OF_STREAM_MARKER = b"\xff\xff\xff\xff\x00\x00\x00\x00"
