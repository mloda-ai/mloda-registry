"""Placeholder license-token text builders (contract: License): the real signed-token format
doesn't exist yet, so the contract accepts plain JSON shaped as
``{"status": "valid" | "expired", "plugins": [<plugin_id>, ...]}``. These are the building blocks
``conformance.BinaryModelConformanceBase``'s license fixtures are built from, covering every state
the contract distinguishes: valid, expired, wrong-plugin, tampered.
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
