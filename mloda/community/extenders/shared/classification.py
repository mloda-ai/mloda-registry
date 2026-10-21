"""Data classification levels and their propagation through a resolved plan."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Mapping
from typing import Any

from mloda.steward import PlanStep

logger = logging.getLogger(__name__)

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
    group (class attribute `masking is True`) that declares its own level caps it at that level."""
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
        reader_rank = None
        if step.reader_data_access is not None:
            reader_rank = _declared(step.reader_data_access[0])
            if reader_rank is not None:
                ranks.append(reader_rank)
        sources = [resolve(source) for source in step.input_feature_edges.get(name, ())]
        if group_rank is not None and getattr(group, MASKING_ATTRIBUTE, None) is True:
            if step.reader_data_access is not None:
                sources.append(reader_rank if reader_rank is not None else len(LEVELS))
            return min(group_rank, max(sources)) if sources else group_rank
        own = max(ranks) if ranks else undeclared_rank
        return max([own, *sources])

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


class RunClassifications:
    """Per-run effective classification levels for an extender; `undeclared=None` turns it off."""

    def __init__(self, undeclared: str | None, owner: str) -> None:
        if undeclared is not None:
            _rank(undeclared, "classification_undeclared")
        self._undeclared = undeclared
        self._owner = owner
        self._runs: dict[str, dict[str, str]] = {}
        self._lock = threading.Lock()

    @property
    def undeclared(self) -> str | None:
        return self._undeclared

    def start(self, run_id: str | None, steps: Iterable[PlanStep], *, raise_on_error: bool) -> None:
        """Store the run's levels; a failure is re-raised when `raise_on_error`, else logged by type and skipped."""
        if self._undeclared is None or run_id is None:
            return
        try:
            levels = feature_classifications(steps, undeclared=self._undeclared)
        except Exception as exc:
            if raise_on_error:
                raise
            logger.warning(
                "%s could not resolve data classification for a run: %s; no level is emitted",
                self._owner,
                type(exc).__name__,
            )
            return
        with self._lock:
            self._runs[run_id] = levels

    def levels(self, run_id: str | None) -> Mapping[str, str]:
        if run_id is None:
            return {}
        with self._lock:
            return self._runs.get(run_id, {})

    def step_level(self, run_id: str | None, feature_names: Iterable[str]) -> str | None:
        """The most restrictive level among the names that have one, else None."""
        levels = self.levels(run_id)
        ranks = [LEVELS.index(levels[name]) for name in feature_names if name in levels]
        return LEVELS[max(ranks)] if ranks else None

    def pop(self, run_id: str | None) -> None:
        if run_id is None:
            return
        with self._lock:
            self._runs.pop(run_id, None)

    def __getstate__(self) -> dict[str, Any]:
        with self._lock:
            return {
                "_undeclared": self._undeclared,
                "_owner": self._owner,
                "_runs": {run: dict(levels) for run, levels in self._runs.items()},
            }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()
