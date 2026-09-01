"""Placeholder license-token text builders (contract: License).

The real signed-token format does not exist yet, so the contract has the binary accept a fixed
placeholder format for now: plain JSON text (not signed) shaped as
``{"status": "valid" | "expired", "plugins": [<plugin_id>, ...]}``. The builders below are the
reusable building blocks ``conformance.BinaryModelConformanceBase``'s license-fixture class
attributes are built from, covering every state the contract's License section distinguishes:
valid, expired, wrong-plugin, and tampered (unparseable text, or valid JSON missing a required
key).
"""

from __future__ import annotations

import json


def license_token_text(status: str, plugins: list[str]) -> str:
    """A placeholder (unsigned) license token: plain JSON shaped as
    ``{"status": ..., "plugins": [...]}`` (contract: License)."""
    return json.dumps({"status": status, "plugins": plugins})


# A tampered token that is not even valid JSON (contract: License).
TAMPERED_UNPARSEABLE_TEXT = "{this is not json"


def tampered_missing_status_text(plugins: list[str]) -> str:
    """Valid JSON missing the required ``status`` key (contract: License)."""
    return json.dumps({"plugins": plugins})


def tampered_missing_plugins_text() -> str:
    """Valid JSON missing the required ``plugins`` key (contract: License)."""
    return json.dumps({"status": "valid"})
