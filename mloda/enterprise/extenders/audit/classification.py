"""Classification policy for AuditExtender: who may request which data classification level."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mloda.community.extenders.shared.classification import LEVELS


class ClassificationDeniedError(RuntimeError):
    """Raised by AuditExtender(classification=...) when a run requests data above the caller's clearance."""


@dataclass(frozen=True)
class ClassificationPolicy:
    """clearance(tenant_id, principal) returns the highest level the caller may read, or None for none (every requested
    feature is then refused); a raise or an unknown level refuses the run as unresolved. undeclared
    is the level assumed for a step that declares nothing. gate_readers (default True) also refuses, at plan time,
    a reader declaring a level above the clearance; False keeps the run-start check only."""

    clearance: Callable[[str | None, str | None], str | None]
    undeclared: str
    gate_readers: bool = True

    def __post_init__(self) -> None:
        if self.undeclared not in LEVELS:
            raise ValueError(f"ClassificationPolicy undeclared must be one of {list(LEVELS)}, got {self.undeclared!r}")
        if not isinstance(self.gate_readers, bool):
            raise TypeError(f"ClassificationPolicy gate_readers must be a bool, got {self.gate_readers!r}")
