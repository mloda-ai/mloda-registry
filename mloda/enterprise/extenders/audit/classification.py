"""Classification policy for AuditExtender: who may request which data classification level."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mloda.provider import BaseInputData

from mloda.community.extenders.shared.classification import LEVELS


class ClassificationDeniedError(RuntimeError):
    """Raised by AuditExtender(classification=...) when a run requests data above the caller's clearance."""


@dataclass(frozen=True)
class ClassificationPolicy:
    """clearance(tenant_id, principal) returns the highest level the caller may read, or None for none (every requested
    feature is then refused); a raise or an unknown level refuses the run as unresolved. undeclared
    is the level assumed for a step that declares nothing. gate_readers (default True) also refuses, at plan time,
    a reader declaring a level above the clearance; False keeps the run-start check only. exempt_readers are reader
    classes, matched exactly (not subclasses), that skip the plan-time gate; the run-start check still applies."""

    clearance: Callable[[str | None, str | None], str | None]
    undeclared: str
    gate_readers: bool = True
    exempt_readers: frozenset[type[BaseInputData]] = frozenset()

    def __post_init__(self) -> None:
        if self.undeclared not in LEVELS:
            raise ValueError(f"ClassificationPolicy undeclared must be one of {list(LEVELS)}, got {self.undeclared!r}")
        if not isinstance(self.gate_readers, bool):
            raise TypeError(f"ClassificationPolicy gate_readers must be a bool, got {self.gate_readers!r}")
        try:
            entries = tuple(self.exempt_readers)
        except TypeError:
            raise TypeError(
                f"ClassificationPolicy exempt_readers must be an iterable of reader classes, got {self.exempt_readers!r}"
            ) from None
        if not all(isinstance(e, type) and issubclass(e, BaseInputData) and e.is_final_reader() for e in entries):
            raise TypeError(f"ClassificationPolicy exempt_readers must hold final reader classes, got {entries!r}")
        object.__setattr__(self, "exempt_readers", frozenset(entries))
        if entries and not self.gate_readers:
            raise ValueError("ClassificationPolicy exempt_readers requires gate_readers=True")
