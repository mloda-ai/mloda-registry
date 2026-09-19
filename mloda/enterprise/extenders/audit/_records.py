"""Record helpers shared by the audit sink (audit_extender.py) and the run-manifest sealer (run_manifest.py)."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _is_blank(value: str | None) -> bool:
    """True when the value is None or blank."""
    return value is None or not value.strip()


def _canonical_json(record: Mapping[str, Any]) -> bytes:
    """The bytes NdjsonAuditSink writes for a record, without the newline."""
    return json.dumps(record, sort_keys=True).encode("utf-8")


def _utc_now() -> str:
    # Audit records require the explicit Z, unlike OpenLineage's +00:00 offset.
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _append_records(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Append one canonical line per record through one O_APPEND descriptor."""
    data = b"".join(_canonical_json(record) + b"\n" for record in records)
    if not data:
        return
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(f"os.write wrote no bytes to {path}")
            view = view[written:]
    finally:
        os.close(fd)
