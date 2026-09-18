"""Entry-point manifest for mloda-enterprise-audit, discovered via the ``mloda.extenders`` entry point."""

from __future__ import annotations

from mloda.steward import Extender

from .audit_extender import AuditExtender

EXTENDERS: list[type[Extender]] = [
    AuditExtender,
]
