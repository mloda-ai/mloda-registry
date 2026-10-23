"""``hmac_sha256`` binary for tests of plugins built on the binary-model contract. Accepts test-signed licenses only; not a real anonymizer.

Run: ``python -m mloda.testing.binary_model.hmac_fake_binary [--plugin-id <id>]``. The id goes in argv because the
mixin runs the binary with a minimal environment. It patches ``simulated_binary`` globals, so run it only as a fresh subprocess."""

from __future__ import annotations

import sys
from typing import Any

from mloda.testing.binary_model import USAGE_ERROR, simulated_binary
from mloda.testing.binary_model.hmac_sha256_reference import (
    KEY_PARAMETER,
    compute_expected_hmac_sha256_column,
    parse_hmac_key,
)
from mloda.testing.binary_model.simulated_binary import install_fake_identity

PLUGIN_ID = "hmac_fake_binary"
OPERATION = "hmac_sha256"
OUTPUT_KEY = "result"
COLUMN_TYPES = frozenset({"utf8"})


def _validate_hmac_parameters(operation: str, parameters: dict[str, Any]) -> None:
    """Only ``hmac_sha256`` is checked; the error text never contains the key value."""
    if operation != OPERATION:
        return
    unknown = set(parameters) - {KEY_PARAMETER}
    if unknown:
        raise simulated_binary._CliError(USAGE_ERROR, f"unknown parameters: {sorted(unknown)}")
    try:
        parse_hmac_key(parameters.get(KEY_PARAMETER))
    except ValueError:
        raise simulated_binary._CliError(USAGE_ERROR, "parameters.key must be a 64-character hex string") from None


def _single_input_column(original: Any) -> Any:
    def validate(config: dict[str, Any]) -> None:
        original(config)
        if config["operation"] == OPERATION and len(config["input_columns"]) != 1:
            raise simulated_binary._CliError(USAGE_ERROR, "hmac_sha256 takes exactly one input column")

    return validate


def _compute_hmac_output(table: Any, config: dict[str, Any]) -> Any:
    import pyarrow as pa

    from mloda.testing.binary_model.arrow_arrays import array_from_values

    values = table.column(config["input_columns"][0]).to_pylist()
    digests = compute_expected_hmac_sha256_column(values, config["parameters"][KEY_PARAMETER])
    written_name = config["output_columns"][OUTPUT_KEY]
    return pa.schema([pa.field(written_name, pa.string())]), [array_from_values(digests, pa.string())]


def _install_hmac_binary_identity(plugin_id: str) -> None:
    install_fake_identity(plugin_id, OPERATION, OUTPUT_KEY, COLUMN_TYPES)
    simulated_binary._validate_hash_parameters = _validate_hmac_parameters
    simulated_binary._validate_config_structure = _single_input_column(simulated_binary._validate_config_structure)
    simulated_binary._compute_hash_output = _compute_hmac_output


def main() -> int:
    plugin_id = PLUGIN_ID
    if len(sys.argv) >= 3 and sys.argv[1] == "--plugin-id":
        plugin_id = sys.argv[2]
        del sys.argv[1:3]
    _install_hmac_binary_identity(plugin_id)
    return simulated_binary.main()


if __name__ == "__main__":
    sys.exit(main())
