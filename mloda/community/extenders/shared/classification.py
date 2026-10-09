"""Data classification levels and their propagation through a resolved plan, shared by the OTel and audit extenders."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mloda.steward import PlanStep

CLASSIFICATION_KEY = "mloda.data.classification"
MASKING_ATTRIBUTE = "masking"
# Least to most restrictive.
LEVELS = ("public", "internal", "confidential", "restricted", "pii")


def _rank(level: Any, owner: str) -> int:
    if level not in LEVELS:
        raise ValueError(f"{owner} has unknown classification level {level!r}; allowed levels are {list(LEVELS)}")
    return LEVELS.index(level)


def _declared(owner: Any) -> int | None:
    """The rank of the level `owner` (a feature group or reader class) declares, else None."""
    level = owner.declared_attributes(None).get(CLASSIFICATION_KEY)
    return None if level is None else _rank(level, owner.__name__)


def feature_classifications(steps: Iterable[PlanStep], *, undeclared: str) -> dict[str, str]:
    """Effective level of every compute output: its declared level or the most restrictive input, unless a masking
    group (class attribute `masking is True`) that declares its own level lowers it to exactly that level."""
    undeclared_rank = _rank(undeclared, "undeclared")
    producers: dict[str, list[PlanStep]] = {}
    for step in steps:
        if step.step_kind == "compute":
            for name in step.feature_names:
                producers.setdefault(name, []).append(step)

    resolved: dict[str, int] = {}
    active: set[str] = set()

    def step_level(step: PlanStep, name: str) -> int:
        group = step.feature_group
        group_rank = None if group is None else _declared(group)
        ranks = [rank for rank in (group_rank,) if rank is not None]
        if step.reader_data_access is not None:
            reader_rank = _declared(step.reader_data_access[0])
            if reader_rank is not None:
                ranks.append(reader_rank)
        if group_rank is not None and getattr(group, MASKING_ATTRIBUTE, None) is True:
            return group_rank
        own = max(ranks) if ranks else undeclared_rank
        return max([own, *(resolve(source) for source in step.input_feature_edges.get(name, ()))])

    def resolve(name: str) -> int:
        if name in resolved:
            return resolved[name]
        if name not in producers:
            raise ValueError(f"feature {name!r} has no producing step in the plan")
        if name in active:
            raise ValueError(f"feature {name!r} depends on itself")
        active.add(name)
        try:
            resolved[name] = max(step_level(step, name) for step in producers[name])
        finally:
            active.discard(name)
        return resolved[name]

    return {name: LEVELS[resolve(name)] for name in producers}
