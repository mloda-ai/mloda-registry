"""Diff two recorded OpenLineage runs (parsed event dicts). Stdlib only; import by module path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_TERMINAL_UNKNOWN = "incomplete"
_STEP_FIELDS = (
    "featureGroupVersion",
    "pluginVersion",
    "computeFramework",
    "inputFeatures",
    "loadedDatasets",
    "schemaFields",
    "rowCount",
    "outcome",
)
_StepKey = tuple[str, "tuple[str, ...] | None", str, int]


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
        run = [c for c in self.changes if c.step is None]
        steps = [c for c in self.changes if c.step is not None and c.field != "step"]
        added_removed = [c for c in self.changes if c.step is not None and c.field == "step"]
        for title, group in (("run", run), ("steps", steps), ("added/removed steps", added_removed)):
            if group:
                lines.append(title + ":")
                lines.extend("  " + _render(c) for c in group)
        lines.append("durations (informational):")
        lines.extend("  " + _render(c) for c in self.durations)
        return "\n".join(lines)


def _show(value: Any) -> str:
    return json.dumps(value, default=str)


def _render(change: Change) -> str:
    where = "" if change.step is None else change.step + ": "
    return f"{where}{change.field}: {_show(change.before)} -> {_show(change.after)}"


def _facets(event: dict[str, Any]) -> dict[str, Any]:
    run = event.get("run")
    facets = run.get("facets") if isinstance(run, dict) else None
    return facets if isinstance(facets, dict) else {}


def _facet(event: dict[str, Any], name: str) -> dict[str, Any]:
    value = _facets(event).get(name)
    return value if isinstance(value, dict) else {}


def _run_id(event: dict[str, Any]) -> Any:
    run = event.get("run")
    return run.get("runId") if isinstance(run, dict) else None


def _parent_run_id(event: dict[str, Any]) -> Any:
    run = _facet(event, "parent").get("run")
    return run.get("runId") if isinstance(run, dict) else None


def _root_ids(events: list[dict[str, Any]]) -> set[Any]:
    named = {_parent_run_id(e) for e in events} - {None}
    return {
        _run_id(e)
        for e in events
        if "mlodaPlan" in _facets(e) or "processingEngine" in _facets(e) or _run_id(e) in named
    }


def read_runs(path: str | Path) -> list[list[dict[str, Any]]]:
    """Group an NDJSON file into runs: each run is its root events then its step events, in file order."""
    events: list[dict[str, Any]] = []
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
    order = list(dict.fromkeys(_run_id(e) for e in events if _run_id(e) in roots))
    return [
        [e for e in events if _run_id(e) == root]
        + [e for e in events if _parent_run_id(e) == root and _run_id(e) != root]
        for root in order
    ]


def _seconds(events: list[dict[str, Any]]) -> float | None:
    terminal = _terminal(events)
    start = next((e for e in events if e.get("eventType") == "START"), None)
    if terminal is None or start is None:
        return None
    try:
        return (_time(terminal) - _time(start)).total_seconds()
    except (ValueError, TypeError, AttributeError):
        return None


def _time(event: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(str(event["eventTime"]).replace("Z", "+00:00"))


def _terminal(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    ended = [e for e in events if e.get("eventType") != "START"]
    return ended[-1] if ended else None


def _outcome(events: list[dict[str, Any]]) -> str:
    terminal = _terminal(events)
    return str(terminal.get("eventType")) if terminal is not None else _TERMINAL_UNKNOWN


@dataclass(frozen=True)
class _Step:
    job: str
    names: tuple[str, ...] | None
    framework: str
    values: dict[str, Any]
    duration: float | None


def _step(events: list[dict[str, Any]]) -> _Step:
    terminal = _terminal(events)
    source = terminal if terminal is not None else events[0]
    step_facet = _facet(source, "mlodaStep")
    inputs = [i for i in (source.get("inputs") or []) if isinstance(i, dict)]
    outputs = [o for o in (source.get("outputs") or []) if isinstance(o, dict)]
    features = step_facet.get("featureNames")
    names: tuple[str, ...] | None
    if isinstance(features, list) and features:
        names = tuple(sorted(str(n) for n in features))
    elif outputs:
        names = tuple(sorted(str(o.get("name")) for o in outputs))
    else:
        names = None
    loaded = [i for i in inputs if "dataSource" in (i.get("facets") or {})]
    plain = [i for i in inputs if i not in loaded]
    first_out = outputs[0] if outputs else {}
    schema = ((first_out.get("facets") or {}).get("schema") or {}).get("fields") or []
    stats = ((first_out.get("outputFacets") or {}).get("outputStatistics")) or {}
    values: dict[str, Any] = {
        "featureGroupVersion": step_facet.get("featureGroupVersion"),
        "pluginVersion": step_facet.get("pluginVersion"),
        "computeFramework": step_facet.get("computeFramework"),
        "inputFeatures": tuple(sorted(str(i.get("name")) for i in plain)),
        "loadedDatasets": tuple(
            sorted(
                (
                    str(i.get("namespace")),
                    str(i.get("name")),
                    bool(((i.get("facets") or {}).get("mlodaDataAccess") or {}).get("identityIsFallback")),
                )
                for i in loaded
            )
        ),
        "schemaFields": tuple((f.get("name"), f.get("type")) for f in schema if isinstance(f, dict)),
        "rowCount": stats.get("rowCount"),
        "outcome": _outcome(events),
    }
    return _Step(
        str(source.get("job", {}).get("name")), names, str(values["computeFramework"] or ""), values, _seconds(events)
    )


def _steps(events: list[dict[str, Any]], roots: set[Any]) -> dict[_StepKey, _Step]:
    by_run: dict[Any, list[dict[str, Any]]] = {}
    for event in events:
        if _run_id(event) not in roots and "mlodaStep" in _facets(event):
            by_run.setdefault(_run_id(event), []).append(event)
    steps = [_step(group) for group in by_run.values()]
    base: dict[tuple[str, tuple[str, ...] | None], list[_Step]] = {}
    for step in steps:
        base.setdefault((step.job, step.names), []).append(step)
    keyed: dict[_StepKey, _Step] = {}
    for (job, names), group in base.items():
        if len(group) == 1:
            keyed[(job, names, "", 0)] = group[0]
            continue
        by_framework: dict[str, list[_Step]] = {}
        for step in group:
            by_framework.setdefault(step.framework, []).append(step)
        for framework, same in by_framework.items():
            for index, step in enumerate(same):
                keyed[(job, names, framework, index + 1 if len(same) > 1 else 0)] = step
    return keyed


def _label(key: _StepKey) -> str:
    job, names, framework, ordinal = key
    label = f"{job} [{'unkeyed' if names is None else ','.join(names)}]"
    if framework:
        label += f" <{framework}>"
    return label + (f" #{ordinal}" if ordinal else "")


def _roots(events: list[dict[str, Any]]) -> tuple[set[Any], list[dict[str, Any]]]:
    ids = _root_ids(events)
    return ids, [e for e in events if _run_id(e) in ids]


def _requested(roots: list[dict[str, Any]]) -> tuple[str, ...] | None:
    for event in roots:
        value = _facet(event, "mlodaPlan").get("requestedFeatures")
        if isinstance(value, list):
            return tuple(sorted(str(v) for v in value))
    return None


def _first(roots: list[dict[str, Any]], facet: str, field: str) -> Any:
    return next((_facet(e, facet)[field] for e in roots if field in _facet(e, facet)), None)


def diff_run_events(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> RunDiff:
    """Compare two runs given as raw event lists; inputs are never mutated."""
    roots_a, root_events_a = _roots(a)
    roots_b, root_events_b = _roots(b)
    job_a = (root_events_a[0].get("job") or {}) if root_events_a else {}
    job_b = (root_events_b[0].get("job") or {}) if root_events_b else {}
    request_a = (job_a.get("namespace"), job_a.get("name"), _requested(root_events_a))
    request_b = (job_b.get("namespace"), job_b.get("name"), _requested(root_events_b))
    both_known = request_a[2] is not None and request_b[2] is not None
    if request_a[:2] != request_b[:2] or (both_known and request_a[2] != request_b[2]):
        return RunDiff(False, (Change(None, "different request", request_a, request_b),), ())

    changes: list[Change] = []
    durations = [Change(None, "duration", _seconds(root_events_a), _seconds(root_events_b))]
    run_values = (
        (
            "structureHash",
            _first(root_events_a, "mlodaPlan", "structureHash"),
            _first(root_events_b, "mlodaPlan", "structureHash"),
        ),
        (
            "mlodaVersion",
            _first(root_events_a, "processingEngine", "version"),
            _first(root_events_b, "processingEngine", "version"),
        ),
        ("outcome", _outcome(root_events_a), _outcome(root_events_b)),
    )
    changes.extend(Change(None, f, x, y) for f, x, y in run_values if x != y)

    steps_a = _steps(a, roots_a)
    steps_b = _steps(b, roots_b)
    for key, step_a in steps_a.items():
        step_b = steps_b.get(key)
        if step_b is None:
            continue
        label = _label(key)
        changes.extend(
            Change(label, f, step_a.values[f], step_b.values[f])
            for f in _STEP_FIELDS
            if step_a.values[f] != step_b.values[f]
        )
        durations.append(Change(label, "duration", step_a.duration, step_b.duration))
    changes.extend(Change(_label(k), "step", "present", "absent") for k in steps_a if k not in steps_b)
    changes.extend(Change(_label(k), "step", "absent", "present") for k in steps_b if k not in steps_a)
    return RunDiff(True, tuple(changes), tuple(durations))
