"""Format of the hand-computed pinning rows of a reference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from mloda.user import DataType

TypedColumns = Mapping[str, tuple[DataType, tuple[Any, ...]]]


@dataclass(frozen=True)
class PinningRow:
    """One hand-computed case: typed input columns, the feature, its context and typed expected columns."""

    case: str
    feature: str
    inputs: TypedColumns
    expected: TypedColumns
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PinningTable:
    """The rows of one operation plus the rules (label to text) its rows pin."""

    operation: str
    rules: Mapping[str, str]
    rows: tuple[PinningRow, ...]
