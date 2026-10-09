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
    is the level assumed for a step that declares nothing."""

    clearance: Callable[[str | None, str | None], str | None]
    undeclared: str

    def __post_init__(self) -> None:
        if self.undeclared not in LEVELS:
            raise ValueError(f"ClassificationPolicy undeclared must be one of {list(LEVELS)}, got {self.undeclared!r}")
