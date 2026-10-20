"""Diff two recorded OpenLineage runs (parsed event dicts). Stdlib only; import by module path."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_TERMINAL_UNKNOWN = "incomplete"
# Field -> the least step progress (0 started, 1 ended, 2 complete) both sides need for it to be comparable.
_STEP_FIELDS = {
    "featureGroupVersion": 0,
    "pluginVersion": 0,
    "computeFramework": 0,
    "inputFeatures": 1,
    "loadedDatasets": 1,
    "schemaFields": 2,
    "rowCount": 2,
    "outcome": 0,
}
_Names = tuple[str, ...] | None
_StepKey = tuple[str, _Names, str, int]


@dataclass(frozen=True)
class Change:
    step: str | None
    field: str
    before: Any
    after: Any


@dataclass(frozen=True)
class RunDiff:
    comparable: bool
    changes: tuple[Change, ...]
    durations: tuple[Change, ...]

    def __str__(self) -> str:
        lines = ["comparable: " + ("yes" if self.comparable else "no")]
        shown = list(self.changes)
        if self.comparable:
            digest = next((c for c in shown if c.step is None and c.field == "structureHash"), None)
            lines.append("structureHash: equal" if digest is None else _render(digest))
            shown = [c for c in shown if c is not digest]
        lines.extend(_render(c) for c in shown)
        if self.durations:
            lines.append("durations (informational):")
            lines.extend("  " + _render(c) for c in self.durations)
        return "\n".join(lines)


def _show(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _render(change: Change) -> str:
    where = "" if change.step is None else change.step + ": "
    return f"{where}{change.field}: {_show(change.before)} -> {_show(change.after)}"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dicts(value: Any) -> list[dict[str, Any]]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def _facets(event: dict[str, Any]) -> dict[str, Any]:
    return _dict(_dict(event.get("run")).get("facets"))


def _facet(event: dict[str, Any], name: str) -> dict[str, Any]:
    return _dict(_facets(event).get(name))


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _run_id(event: dict[str, Any]) -> str | None:
    return _str_or_none(_dict(event.get("run")).get("runId"))


def _parent_run_id(event: dict[str, Any]) -> str | None:
    return _str_or_none(_dict(_facet(event, "parent").get("run")).get("runId"))


def _root_ids(events: list[dict[str, Any]]) -> set[str]:
    named = {p for p in map(_parent_run_id, events) if p is not None}
    return {
        rid
        for e in events
        if (rid := _run_id(e)) is not None
        and ("mlodaPlan" in _facets(e) or "processingEngine" in _facets(e) or rid in named)
    }


def read_runs(*paths: str | Path) -> list[list[dict[str, Any]]]:
    """Group NDJSON files, read in the given order, into runs: each run is its root events then its step events."""
    events: list[dict[str, Any]] = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                events.append(event)
    roots = _root_ids(events)
    own: dict[str, list[dict[str, Any]]] = {}
    children: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        rid, parent = _run_id(event), _parent_run_id(event)
        if rid in roots and rid is not None:
            own.setdefault(rid, []).append(event)
        if parent in roots and parent is not None and rid != parent:
            children.setdefault(parent, []).append(event)
    return [own[root] + children.get(root, []) for root in own]


def _time(event: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(str(event["eventTime"]).replace("Z", "+00:00"))


def _terminal(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    ended = [e for e in events if e.get("eventType") != "START"]
    return ended[-1] if ended else None


def _seconds(events: list[dict[str, Any]]) -> float | None:
    terminal = _terminal(events)
    start = next((e for e in events if e.get("eventType") == "START"), None)
    if terminal is None or start is None:
        return None
    try:
        return (_time(terminal) - _time(start)).total_seconds()
    except (ValueError, TypeError, AttributeError, KeyError):
        return None


def _outcome(events: list[dict[str, Any]]) -> str:
    terminal = _terminal(events)
    return str(terminal.get("eventType")) if terminal is not None else _TERMINAL_UNKNOWN


@dataclass(frozen=True)
class _Step:
    job: str
    declared: _Names
    outputs: _Names
    framework: str
    values: dict[str, Any]
    level: int
    duration: float | None

    def base(self, legacy: bool) -> tuple[str, _Names]:
        return self.job, self.outputs if legacy else self.declared or self.outputs


def _sorted_names(values: list[Any]) -> _Names:
    return tuple(sorted(str(v) for v in values)) or None


def _schema_fields(output: dict[str, Any]) -> tuple[Any, ...]:
    fields = _dict(_dict(output.get("facets")).get("schema")).get("fields")
    return tuple((f.get("name"), f.get("type")) for f in _dicts(fields))


def _loaded_dataset(item: dict[str, Any]) -> tuple[str, str, bool]:
    access = _dict(_dict(item.get("facets")).get("mlodaDataAccess"))
    return str(item.get("namespace")), str(item.get("name")), bool(access.get("identityIsFallback"))


def _step(events: list[dict[str, Any]]) -> _Step:
    terminal = _terminal(events)
    source = terminal if terminal is not None else events[0]
    step_facet = _facet(source, "mlodaStep")
    inputs = _dicts(source.get("inputs"))
    outputs = _dicts(source.get("outputs"))
    loaded = [i for i in inputs if "dataSource" in _dict(i.get("facets"))]
    plain = [i for i in inputs if i not in loaded]
    first_out = outputs[0] if outputs else {}
    stats = _dict(_dict(first_out.get("outputFacets")).get("outputStatistics"))
    features = step_facet.get("featureNames")
    values: dict[str, Any] = {
        "featureGroupVersion": step_facet.get("featureGroupVersion"),
        "pluginVersion": step_facet.get("pluginVersion"),
        "computeFramework": step_facet.get("computeFramework"),
        "inputFeatures": tuple(sorted(str(i.get("name")) for i in plain)),
        "loadedDatasets": tuple(sorted(_loaded_dataset(i) for i in loaded)),
        "schemaFields": _schema_fields(first_out),
        "rowCount": stats.get("rowCount"),
        "outcome": _outcome(events),
    }
    return _Step(
        str(_dict(source.get("job")).get("name")),
        _sorted_names(features) if isinstance(features, list) else None,
        _sorted_names([o.get("name") for o in outputs]),
        str(values["computeFramework"] or ""),
        values,
        {"incomplete": 0, "COMPLETE": 2}.get(values["outcome"], 1),
        _seconds(events),
    )


def _steps(events: list[dict[str, Any]], roots: set[str]) -> list[_Step]:
    by_run: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        rid = _run_id(event)
        if rid is not None and rid not in roots and "mlodaStep" in _facets(event):
            by_run.setdefault(rid, []).append(event)
    return [_step(group) for group in by_run.values()]


def _duplicates(steps: list[_Step], legacy: bool) -> tuple[set[Any], set[Any]]:
    """Base keys and (base key, framework) groups holding more than one step."""
    bases = Counter(s.base(legacy) for s in steps)
    groups = Counter((s.base(legacy), s.framework) for s in steps)
    return {k for k, n in bases.items() if n > 1}, {k for k, n in groups.items() if n > 1}


def _keyed(steps: list[_Step], legacy: bool, bases: set[Any], groups: set[Any]) -> dict[_StepKey, _Step]:
    keyed: dict[_StepKey, _Step] = {}
    seen: Counter[Any] = Counter()
    for step in steps:
        base = step.base(legacy)
        if base not in bases:
            keyed[(*base, "", 0)] = step
            continue
        group = (base, step.framework)
        seen[group] += 1
        keyed[(*base, step.framework, seen[group] if group in groups else 0)] = step
    return keyed


def _key_both(a: list[_Step], b: list[_Step]) -> tuple[dict[_StepKey, _Step], dict[_StepKey, _Step]]:
    legacy = any(s.declared for s in a) != any(s.declared for s in b)
    (bases_a, groups_a), (bases_b, groups_b) = _duplicates(a, legacy), _duplicates(b, legacy)
    bases, groups = bases_a | bases_b, groups_a | groups_b
    return _keyed(a, legacy, bases, groups), _keyed(b, legacy, bases, groups)


def _label(key: _StepKey) -> str:
    job, names, framework, ordinal = key
    label = f"{job} [{'unkeyed' if names is None else ','.join(names)}]"
    if framework:
        label += f" <{framework}>"
    return label + (f" #{ordinal}" if ordinal else "")


def _roots(events: list[dict[str, Any]]) -> tuple[set[str], list[dict[str, Any]]]:
    ids = _root_ids(events)
    if len(ids) > 1:
        raise ValueError(f"expected one root run per argument, found {len(ids)}; split the events with read_runs")
    return ids, [e for e in events if _run_id(e) in ids]


def _requested(roots: list[dict[str, Any]]) -> tuple[str, ...] | None:
    for event in roots:
        value = _facet(event, "mlodaPlan").get("requestedFeatures")
        if isinstance(value, list):
            return tuple(sorted(str(v) for v in value))
    return None


def _first(roots: list[dict[str, Any]], facet: str, field: str) -> Any:
    return next((v for e in roots if (v := _facet(e, facet).get(field)) is not None), None)


def _step_changes(label: str, a: _Step, b: _Step) -> list[Change]:
    level = min(a.level, b.level)
    return [
        Change(label, f, a.values[f], b.values[f])
        for f, needed in _STEP_FIELDS.items()
        if level >= needed and a.values[f] != b.values[f]
    ]


def diff_run_events(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> RunDiff:
    """Compare two runs given as raw event lists; inputs are never mutated."""
    roots_a, root_events_a = _roots(a)
    roots_b, root_events_b = _roots(b)
    job_a = _dict(root_events_a[0].get("job")) if root_events_a else {}
    job_b = _dict(root_events_b[0].get("job")) if root_events_b else {}
    request_a = (job_a.get("namespace"), job_a.get("name"), _requested(root_events_a))
    request_b = (job_b.get("namespace"), job_b.get("name"), _requested(root_events_b))
    both_known = request_a[2] is not None and request_b[2] is not None
    if request_a[:2] != request_b[:2] or (both_known and request_a[2] != request_b[2]):
        return RunDiff(False, (Change(None, "different request", request_a, request_b),), ())

    durations = [Change(None, "duration", _seconds(root_events_a), _seconds(root_events_b))]
    changes: list[Change] = []
    for field, facet, name in (
        ("structureHash", "mlodaPlan", "structureHash"),
        ("mlodaVersion", "processingEngine", "version"),
    ):
        before, after = _first(root_events_a, facet, name), _first(root_events_b, facet, name)
        if before != after:
            changes.append(Change(None, field, before, after))

    steps_a, steps_b = _key_both(_steps(a, roots_a), _steps(b, roots_b))
    for key, step_a in steps_a.items():
        step_b = steps_b.get(key)
        if step_b is not None:
            changes.extend(_step_changes(_label(key), step_a, step_b))
            durations.append(Change(_label(key), "duration", step_a.duration, step_b.duration))
    changes.extend(Change(_label(k), "step", "present", "absent") for k in steps_a if k not in steps_b)
    changes.extend(Change(_label(k), "step", "absent", "present") for k in steps_b if k not in steps_a)
    outcome_a, outcome_b = _outcome(root_events_a), _outcome(root_events_b)
    if outcome_a != outcome_b:
        changes.append(Change(None, "outcome", outcome_a, outcome_b))
    return RunDiff(True, tuple(changes), tuple(durations))
