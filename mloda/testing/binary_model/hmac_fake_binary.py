"""Fake ``hmac_sha256`` binary for tests only; accepts test-signed licenses only.

Run: ``python -m mloda.testing.binary_model.hmac_fake_binary [--plugin-id <id>]``. The id is argv because the mixin
passes a minimal environment."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from mloda.testing.binary_model import simulated_binary
from mloda.testing.binary_model.hmac_sha256_reference import (
    KEY_PARAMETER,
    compute_expected_hmac_sha256_column,
    parse_hmac_key,
)
from mloda.testing.binary_model.simulated_binary import FakeBinary, FakeOperation, FakeUsageError

if TYPE_CHECKING:
    import pyarrow as pa

PLUGIN_ID = "hmac_fake_binary"
OPERATION = "hmac_sha256"
OUTPUT_KEY = "result"
COLUMN_TYPES = frozenset({"utf8"})


def _validate_hmac_config(config: dict[str, Any]) -> None:
    """The error text never contains the key value."""
    if len(config["input_columns"]) != 1:
        raise FakeUsageError("hmac_sha256 takes exactly one input column")
    parameters = config["parameters"]
    unknown = set(parameters) - {KEY_PARAMETER}
    if unknown:
        raise FakeUsageError(f"unknown parameters: {sorted(unknown)}")
    try:
        parse_hmac_key(parameters.get(KEY_PARAMETER))
    except ValueError:
        raise FakeUsageError("parameters.key must be a 64-character hex string") from None


def _compute_hmac_output(table: pa.Table, config: dict[str, Any]) -> tuple[pa.Schema, list[pa.Array]]:
    import pyarrow as pa

    from mloda.testing.binary_model.arrow_arrays import array_from_values

    values = table.column(config["input_columns"][0]).to_pylist()
    digests = compute_expected_hmac_sha256_column(values, config["parameters"][KEY_PARAMETER])
    written_name = config["output_columns"][OUTPUT_KEY]
    return pa.schema([pa.field(written_name, pa.string())]), [array_from_values(digests, pa.string())]


HMAC_OPERATION = FakeOperation(OPERATION, (OUTPUT_KEY,), _compute_hmac_output, _validate_hmac_config)


def main() -> int:
    plugin_id = PLUGIN_ID
    if len(sys.argv) >= 3 and sys.argv[1] == "--plugin-id":
        plugin_id = sys.argv[2]
        del sys.argv[1:3]
    return simulated_binary.main(FakeBinary(plugin_id, (HMAC_OPERATION,), COLUMN_TYPES))


if __name__ == "__main__":
    sys.exit(main())
