"""Tenant and project identity read from a lifecycle or hook context."""

from __future__ import annotations

from typing import Any


def _present(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def tenant_identity(context: Any) -> dict[str, str]:
    """Non-blank tenant_id and project_id of the context, keyed by field name; never the principal."""
    pairs = (
        ("tenant_id", _present(getattr(context, "tenant_id", None))),
        ("project_id", _present(getattr(context, "project_id", None))),
    )
    return {key: value for key, value in pairs if value is not None}
