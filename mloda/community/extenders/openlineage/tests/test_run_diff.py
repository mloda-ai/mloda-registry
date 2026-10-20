"""Tests for the run record diff: read_runs groups an NDJSON file into runs, diff_run_events compares two runs.

Runs come from real OpenLineageExtender recordings; variants are mutated deep copies of them."""

from __future__ import annotations

import copy
import json
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from mloda.community.extenders.openlineage.openlineage_extender import OpenLineageExtender
from mloda.community.extenders.openlineage.run_diff import Change, RunDiff, diff_run_events, read_runs
from mloda.testing.extenders.openlineage import make_recording_client
from mloda.testing.extenders.runners import run_two_features
from openlineage.client.serde import Serde

Events = list[dict[str, Any]]
Edit = Callable[[dict[str, Any]], object]

_ROOT = "mloda.run_all"
_PLUS_ONE = "MlodaTestingValueIntPlusOne"
_SENTINEL = "SENTINEL-must-not-leak"


def _record(**options: Any) -> Events:
    client, transport = make_recording_client()
    run_two_features(OpenLineageExtender(client=client, **options))
    return [json.loads(Serde.to_json(event)) for event in transport.events]


@pytest.fixture(scope="module")
def recorded() -> Events:
    return _record()


def _variant(
    events: Events, edit: Edit, *, job: str | None = None, kind: str | None = None, run_id: str | None = None
) -> Events:
    """A deep copy of events with edit applied to those matching the job suffix, event type and run id."""
    copied = copy.deepcopy(events)
    for event in copied:
        if (
            (job is None or event["job"]["name"].endswith(job))
            and (kind is None or event["eventType"] == kind)
            and (run_id is None or event["run"]["runId"] == run_id)
        ):
            edit(event)
    return copied


def _facets(event: dict[str, Any]) -> dict[str, Any]:
    facets: dict[str, Any] = event["run"]["facets"]
    return facets


def _split(events: Events, root_job: str = _ROOT) -> tuple[Events, Events]:
    return [e for e in events if e["job"]["name"] == root_job], [e for e in events if e["job"]["name"] != root_job]


def _changes(changes: tuple[Change, ...]) -> dict[str, Change]:
    found = {change.field: change for change in changes}
    assert len(found) == len(changes)
    return found


def _with_duplicate(events: Events, framework: str | None) -> tuple[Events, str]:
    """events plus a second copy of the plus-one step under a new run id (optionally on another framework)."""
    run_id = str(uuid.uuid4())
    copies = [copy.deepcopy(e) for e in events if e["job"]["name"].endswith(_PLUS_ONE)]
    for event in copies:
        event["run"]["runId"] = run_id
        if framework is not None:
            _facets(event)["mlodaStep"]["computeFramework"] = framework
    return events + copies, run_id


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n")
    return path


def _dumps(events: Events) -> list[str]:
    return [json.dumps(event) for event in events]


def _add_dataset_input(fallback: bool) -> Edit:
    def edit(event: dict[str, Any]) -> None:
        event["inputs"][0]["facets"] = {
            "dataSource": {"name": "csv://a", "uri": "csv://a"},
            "mlodaDataAccess": {"identityIsFallback": fallback},
        }

    return edit


def _shift(seconds: float) -> Edit:
    def edit(event: dict[str, Any]) -> None:
        event["eventTime"] = (datetime.fromisoformat(event["eventTime"]) + timedelta(seconds=seconds)).isoformat()

    return edit


class TestRunDiffIdenticalRuns:
    def test_two_identical_runs_are_comparable_without_changes(self, recorded: Events) -> None:
        other = _record()
        snapshots = copy.deepcopy((recorded, other))

        diff = diff_run_events(recorded, other)

        assert (recorded, other) == snapshots

        assert isinstance(diff, RunDiff)
        assert diff.comparable is True
        assert diff.changes == ()
        assert diff.durations
        assert all(isinstance(change, Change) for change in diff.durations)
        assert "structureHash: equal" in str(diff)

    def test_durations_are_informational_and_never_changes(self, recorded: Events) -> None:
        slower = _variant(recorded, _shift(3600), job=_ROOT, kind="COMPLETE")

        diff = diff_run_events(recorded, slower)

        assert diff.comparable is True
        assert diff.changes == ()
        run_level = [d for d in diff.durations if d.step is None]
        assert len(run_level) == 1
        assert all(d.step is not None for d in diff.durations if d is not run_level[0])
        run_duration = run_level[0]
        assert run_duration.field == "duration"
        assert run_duration.before != run_duration.after


class TestRunDiffRunLevelChanges:
    def test_structure_hash_change(self, recorded: Events) -> None:
        other = _variant(recorded, lambda e: _facets(e)["mlodaPlan"].update(structureHash="f" * 64), job=_ROOT)

        found = _changes(diff_run_events(recorded, other).changes)

        assert set(found) == {"structureHash"}
        assert found["structureHash"].step is None
        assert found["structureHash"].after == "f" * 64

    def test_mloda_version_change(self, recorded: Events) -> None:
        other = _variant(recorded, lambda e: _facets(e)["processingEngine"].update(version="99.0.0"), job=_ROOT)

        found = _changes(diff_run_events(recorded, other).changes)

        assert set(found) == {"mlodaVersion"}
        assert found["mlodaVersion"].step is None
        assert found["mlodaVersion"].after == "99.0.0"

    def test_run_without_a_terminal_event_is_incomplete(self, recorded: Events) -> None:
        other = [e for e in recorded if not (e["job"]["name"] == _ROOT and e["eventType"] != "START")]

        found = _changes(diff_run_events(recorded, other).changes)

        assert set(found) == {"outcome"}
        assert found["outcome"].step is None
        assert (found["outcome"].before, found["outcome"].after) == ("COMPLETE", "incomplete")


_RECORDED = object()
_PLUS_ONE_SCHEMA = (("MlodaTestingValueIntPlusOne", "int"),)
_STEP_CHANGES: list[Any] = [
    pytest.param(
        None,
        lambda e: _facets(e)["mlodaStep"].update(featureGroupVersion="v2"),
        {"featureGroupVersion": (_RECORDED, "v2")},
        id="featureGroupVersion",
    ),
    pytest.param(
        None,
        lambda e: _facets(e)["mlodaStep"].update(pluginVersion="p2"),
        {"pluginVersion": (_RECORDED, "p2")},
        id="pluginVersion",
    ),
    pytest.param(
        None,
        lambda e: _facets(e)["mlodaStep"].update(computeFramework="OtherFramework"),
        {"computeFramework": ("PyArrowTable", "OtherFramework")},
        id="computeFramework",
    ),
    pytest.param(
        "COMPLETE",
        lambda e: e["inputs"][0].update(name="other_input"),
        {"inputFeatures": (("value_int",), ("other_input",))},
        id="inputFeatures",
    ),
    pytest.param(
        "COMPLETE",
        _add_dataset_input(False),
        {
            "inputFeatures": (("value_int",), ()),
            "loadedDatasets": ((), (("mloda", "value_int", False),)),
        },
        id="loadedDatasets",
    ),
    pytest.param(
        "COMPLETE",
        lambda e: e["outputs"][0]["facets"]["schema"]["fields"][0].update(type="string"),
        {"schemaFields": (_PLUS_ONE_SCHEMA, (("MlodaTestingValueIntPlusOne", "string"),))},
        id="schemaFields",
    ),
    pytest.param(
        "COMPLETE",
        lambda e: e["outputs"][0]["outputFacets"]["outputStatistics"].update(rowCount=99),
        {"rowCount": (12, 99)},
        id="rowCount",
    ),
    pytest.param("COMPLETE", lambda e: e.update(eventType="FAIL"), {"outcome": ("COMPLETE", "FAIL")}, id="outcome"),
]


class TestRunDiffStepChanges:
    @pytest.mark.parametrize(("kind", "edit", "expected"), _STEP_CHANGES)
    def test_each_step_field_change_is_reported_for_that_step(
        self, recorded: Events, kind: str | None, edit: Edit, expected: dict[str, tuple[Any, Any]]
    ) -> None:
        other = _variant(recorded, edit, job=_PLUS_ONE, kind=kind)
        snapshots = copy.deepcopy((recorded, other))

        found = _changes(diff_run_events(recorded, other).changes)

        assert (recorded, other) == snapshots
        assert set(found) == set(expected)
        recorded_step = _facets(next(e for e in recorded if e["job"]["name"].endswith(_PLUS_ONE)))["mlodaStep"]
        for field, (before, after) in expected.items():
            change = found[field]
            assert change.step is not None
            assert _PLUS_ONE in str(change.step)
            assert (change.before, change.after) == (recorded_step.get(field) if before is _RECORDED else before, after)

    def test_a_flipped_identity_fallback_flag_is_a_loaded_dataset_change(self, recorded: Events) -> None:
        before = _variant(recorded, _add_dataset_input(False), job=_PLUS_ONE, kind="COMPLETE")
        after = _variant(recorded, _add_dataset_input(True), job=_PLUS_ONE, kind="COMPLETE")

        found = _changes(diff_run_events(before, after).changes)

        assert set(found) == {"loadedDatasets"}
        assert "false" in json.dumps(found["loadedDatasets"].before)
        assert "true" in json.dumps(found["loadedDatasets"].after)

    def test_a_step_without_a_terminal_event_is_incomplete(self, recorded: Events) -> None:
        other = [e for e in recorded if not (e["job"]["name"].endswith(_PLUS_ONE) and e["eventType"] == "COMPLETE")]

        found = _changes(diff_run_events(recorded, other).changes)

        assert set(found) == {"outcome"}
        assert (found["outcome"].before, found["outcome"].after) == ("COMPLETE", "incomplete")
        assert _PLUS_ONE in str(found["outcome"].step)

    def test_a_failed_step_without_outputs_is_still_keyed_by_its_feature_names(self, recorded: Events) -> None:
        failed = _variant(
            recorded, lambda e: e.update(eventType="FAIL", inputs=[], outputs=[]), job=_PLUS_ONE, kind="COMPLETE"
        )

        found = _changes(diff_run_events(recorded, failed).changes)

        assert "step" not in found
        assert not {"schemaFields", "rowCount"} & set(found)
        assert (found["outcome"].before, found["outcome"].after) == ("COMPLETE", "FAIL")
        assert _PLUS_ONE in str(found["outcome"].step)


class TestRunDiffStepKeys:
    def test_a_step_event_without_feature_names_falls_back_to_its_output_names(self, recorded: Events) -> None:
        stripped = _variant(recorded, lambda e: _facets(e)["mlodaStep"].pop("featureNames", None), job=_PLUS_ONE)

        assert diff_run_events(recorded, stripped).changes == ()

    def test_a_step_with_neither_feature_names_nor_outputs_is_reported_as_unkeyed(self, recorded: Events) -> None:
        def strip(event: dict[str, Any]) -> None:
            _facets(event)["mlodaStep"].pop("featureNames", None)
            event["outputs"] = []

        stripped = _variant(recorded, strip, job=_PLUS_ONE)

        assert "unkeyed" in str(diff_run_events(recorded, stripped))

    @pytest.mark.parametrize("framework", [None, "OtherFramework"], ids=["ordinal", "framework"])
    def test_a_repeated_key_is_disambiguated_deterministically(self, recorded: Events, framework: str | None) -> None:
        before, _ = _with_duplicate(recorded, framework)
        after, duplicate_id = _with_duplicate(recorded, framework)

        assert diff_run_events(before, after).changes == ()

        changed = _variant(
            after,
            lambda e: e["outputs"][0]["outputFacets"]["outputStatistics"].update(rowCount=99),
            kind="COMPLETE",
            run_id=duplicate_id,
        )
        found = _changes(diff_run_events(before, changed).changes)

        assert set(found) == {"rowCount"}
        assert diff_run_events(before, changed) == diff_run_events(before, changed)

    @pytest.mark.parametrize("extra_on_after", [True, False], ids=["added", "removed"])
    def test_an_extra_step_on_another_framework_leaves_the_original_step_unchanged(
        self, recorded: Events, extra_on_after: bool
    ) -> None:
        with_extra, _ = _with_duplicate(recorded, "OtherFramework")
        before, after = (recorded, with_extra) if extra_on_after else (with_extra, recorded)

        found = _changes(diff_run_events(before, after).changes)

        assert set(found) == {"step"}
        assert "OtherFramework" in str(found["step"].step)
        assert (found["step"].before, found["step"].after) == (
            ("absent", "present") if extra_on_after else ("present", "absent")
        )

    @pytest.mark.parametrize("new_is_first", [True, False], ids=["new-vs-old", "old-vs-new"])
    def test_a_record_with_feature_names_diffs_against_one_without_them_by_complete_output_names(
        self, recorded: Events, new_is_first: bool
    ) -> None:
        def requested_only(event: dict[str, Any]) -> None:
            step = _facets(event).get("mlodaStep")
            if step is not None:
                step["featureNames"] = ["requested_only"]

        def legacy(event: dict[str, Any]) -> None:
            step = _facets(event).get("mlodaStep")
            if step is not None:
                step.pop("featureNames", None)

        new = _variant(recorded, requested_only)
        old = _variant(recorded, legacy)
        before, after = (new, old) if new_is_first else (old, new)

        assert diff_run_events(before, after).changes == ()

    def test_a_validation_run_event_without_a_step_facet_is_skipped(self, recorded: Events) -> None:
        extra = [copy.deepcopy(e) for e in recorded if e["job"]["name"].endswith(_PLUS_ONE)]
        validation_id = str(uuid.uuid4())
        for event in extra:
            del _facets(event)["mlodaStep"]
            event["job"]["name"] = "validation.job"
            event["run"]["runId"] = validation_id

        assert diff_run_events(recorded, recorded + extra).changes == ()


class TestRunDiffAddedAndRemovedSteps:
    @pytest.mark.parametrize("removed", [True, False], ids=["removed", "added"])
    def test_a_step_missing_on_one_side_is_reported(self, recorded: Events, removed: bool) -> None:
        gone_run_ids = {e["run"]["runId"] for e in recorded if e["job"]["name"].endswith(_PLUS_ONE)}
        smaller = [e for e in recorded if e["run"]["runId"] not in gone_run_ids]
        before, after = (recorded, smaller) if removed else (smaller, recorded)

        found = _changes(diff_run_events(before, after).changes)

        assert set(found) == {"step"}
        assert _PLUS_ONE in str(found["step"].step)
        assert (found["step"].before, found["step"].after) == (
            ("present", "absent") if removed else ("absent", "present")
        )


class TestRunDiffComparability:
    @pytest.mark.parametrize(
        "edit",
        [
            lambda e: _facets(e)["mlodaPlan"].update(requestedFeatures=["something_else"]),
            lambda e: e["job"].update(name="renamed.root"),
            lambda e: e["job"].update(namespace="other-ns"),
        ],
        ids=["requested-features", "job-name", "job-namespace"],
    )
    def test_a_different_request_is_not_comparable_and_compares_no_steps(self, recorded: Events, edit: Edit) -> None:
        other = _variant(recorded, edit, job=_ROOT)
        other = _variant(
            other,
            lambda e: e["outputs"][0]["outputFacets"]["outputStatistics"].update(rowCount=99),
            job=_PLUS_ONE,
            kind="COMPLETE",
        )

        diff = diff_run_events(recorded, other)

        assert diff.comparable is False
        assert [change.field for change in diff.changes] == ["different request"]
        assert diff.changes[0].step is None

    def test_a_side_without_requested_features_is_unknown_and_still_comparable(self, recorded: Events) -> None:
        root = _split(recorded)[0][0]
        assert "requestedFeatures" in _facets(root)["mlodaPlan"]
        legacy = _variant(recorded, lambda e: _facets(e)["mlodaPlan"].pop("requestedFeatures", None), job=_ROOT)

        diff = diff_run_events(recorded, legacy)

        assert diff.comparable is True
        assert diff.changes == ()


class TestRunDiffReport:
    def test_the_report_orders_run_changes_step_changes_and_durations(self, recorded: Events) -> None:
        other = _variant(recorded, lambda e: _facets(e)["mlodaPlan"].update(structureHash="f" * 64), job=_ROOT)
        other = _variant(other, lambda e: _facets(e)["processingEngine"].update(version="99.0.0"), job=_ROOT)
        other = _variant(other, lambda e: e.update(eventType="FAIL"), job=_ROOT, kind="COMPLETE")
        other = _variant(
            other,
            lambda e: e["outputs"][0]["outputFacets"]["outputStatistics"].update(rowCount=99),
            job=_PLUS_ONE,
            kind="COMPLETE",
        )
        other, _ = _with_duplicate(other, "OtherFramework")

        diff = diff_run_events(recorded, other)
        report = str(diff)

        assert [(c.step is None, c.field) for c in diff.changes] == [
            (True, "structureHash"),
            (True, "mlodaVersion"),
            (False, "rowCount"),
            (False, "step"),
            (True, "outcome"),
        ]
        needles = ("comparable:", "structureHash", "mlodaVersion", "rowCount", "step: absent", "outcome", "durations")
        positions = [report.index(text) for text in needles]
        assert positions == sorted(positions)

    def test_an_unchanged_structure_hash_is_stated_in_the_report(self, recorded: Events) -> None:
        other = _variant(
            recorded,
            lambda e: e["outputs"][0]["outputFacets"]["outputStatistics"].update(rowCount=99),
            job=_PLUS_ONE,
            kind="COMPLETE",
        )

        lines = str(diff_run_events(recorded, other)).splitlines()

        assert lines[0] == "comparable: yes"
        assert lines[1] == "structureHash: equal"

    def test_a_not_comparable_report_has_no_durations_section(self, recorded: Events) -> None:
        other = _variant(recorded, lambda e: e["job"].update(name="renamed.root"), job=_ROOT)

        diff = diff_run_events(recorded, other)

        assert diff.durations == ()
        assert "durations" not in str(diff)

    def test_unread_event_content_never_reaches_the_report(self, recorded: Events) -> None:
        def plant(event: dict[str, Any]) -> None:
            facets = event["run"]["facets"]
            if event["job"]["name"] == _ROOT:
                facets["parent"] = {
                    "run": {"runId": str(uuid.uuid4())},
                    "job": {"namespace": _SENTINEL, "name": _SENTINEL},
                }
            else:
                facets["errorMessage"] = {
                    "message": _SENTINEL,
                    "programmingLanguage": "python",
                    "stackTrace": _SENTINEL,
                }
            event["job"]["facets"] = {"ownership": {"owners": [{"name": _SENTINEL, "type": "team"}]}}
            for output in event["outputs"]:
                output["facets"]["mlodaDeclared"]["attributes"] = {"note": _SENTINEL}
                output["facets"]["ownership"] = {"owners": [{"name": _SENTINEL, "type": "team"}]}
                output["facets"]["tags"] = {"tags": [{"key": "mloda.classification", "value": _SENTINEL}]}

        planted = _variant(recorded, plant)
        planted = _variant(
            planted,
            lambda e: e["outputs"][0]["outputFacets"]["outputStatistics"].update(rowCount=99),
            job=_PLUS_ONE,
            kind="COMPLETE",
        )

        diff = diff_run_events(recorded, planted)

        assert set(_changes(diff.changes)) == {"rowCount"}
        assert _SENTINEL not in str(diff)
        assert _SENTINEL not in repr(diff)


class TestRunDiffOddShapes:
    @pytest.mark.parametrize(
        "edit",
        [
            lambda e: e["run"].update(runId=["unhashable"]),
            lambda e: _facets(e).update(parent={"run": {"runId": {"a": 1}}, "job": {}}),
            lambda e: e["run"].update(runId=7),
        ],
        ids=["list-run-id", "dict-parent-run-id", "int-run-id"],
    )
    def test_a_non_string_run_id_is_ignored_without_raising(self, recorded: Events, tmp_path: Path, edit: Edit) -> None:
        junk = _variant(recorded, edit, job=_PLUS_ONE, kind="START")
        extra = [e for e, o in zip(junk, recorded, strict=True) if e != o]
        assert extra

        assert diff_run_events(recorded, recorded + extra).changes == ()
        path = _write(tmp_path / "events.ndjson", _dumps(recorded + extra))
        assert len(read_runs(path)) == 1

    def test_an_event_without_a_run_id_is_never_a_root(self, recorded: Events, tmp_path: Path) -> None:
        junk = {"eventType": "START", "job": {"name": "x"}, "run": {"facets": {"mlodaPlan": {}}}}
        path = _write(tmp_path / "events.ndjson", _dumps([*recorded, junk]))

        assert len(read_runs(path)) == 1
        assert diff_run_events(recorded, [*recorded, junk]).changes == ()

    def test_an_event_without_event_time_has_no_duration(self, recorded: Events) -> None:
        other = _variant(recorded, lambda e: e.pop("eventTime"), job=_ROOT, kind="COMPLETE")

        diff = diff_run_events(recorded, other)

        run_level = [d for d in diff.durations if d.step is None]
        assert run_level[0].after is None
        assert run_level[0].before is not None
        assert diff.changes == ()

    @pytest.mark.parametrize(
        "edit",
        [
            lambda e: e.update(job=None),
            lambda e: e["run"].update(facets="not-a-dict"),
            lambda e: e["outputs"][0].update(facets="not-a-dict"),
            lambda e: e["outputs"][0]["facets"].update(schema="not-a-dict"),
            lambda e: e["outputs"][0]["outputFacets"].update(outputStatistics=[1]),
            lambda e: e["outputs"][0].update(outputFacets="not-a-dict"),
            lambda e: e.update(inputs=["junk", 3, None]),
            lambda e: e["outputs"][0]["facets"]["schema"].update(fields="oops"),
        ],
        ids=[
            "job-null",
            "run-facets",
            "output-facets",
            "schema-facet",
            "output-statistics",
            "output-facets-field",
            "input-items",
            "schema-fields",
        ],
    )
    def test_a_valid_json_event_of_an_odd_shape_never_raises(self, recorded: Events, edit: Edit) -> None:
        odd = _variant(recorded, edit, job=_PLUS_ONE, kind="COMPLETE")

        assert isinstance(diff_run_events(recorded, odd), RunDiff)
        assert isinstance(diff_run_events(odd, recorded), RunDiff)


class TestRunDiffRootHandling:
    def test_more_than_one_root_run_is_rejected(self, recorded: Events) -> None:
        both = recorded + _record()

        with pytest.raises(ValueError):
            diff_run_events(both, recorded)
        with pytest.raises(ValueError):
            diff_run_events(recorded, both)

    @pytest.mark.parametrize("nulled", ["START", "COMPLETE"])
    def test_an_explicit_null_on_one_root_event_never_masks_the_other_root_event(
        self, recorded: Events, nulled: str
    ) -> None:
        other = _variant(recorded, lambda e: _facets(e)["mlodaPlan"].update(structureHash=None), job=_ROOT, kind=nulled)

        assert diff_run_events(recorded, other).changes == ()
        assert diff_run_events(other, recorded).changes == ()


class TestReadRuns:
    @pytest.mark.parametrize("per_line_files", [False, True], ids=["one_file", "file_per_line"])
    def test_two_runs_are_grouped_with_blank_and_truncated_lines_skipped(
        self, recorded: Events, tmp_path: Path, per_line_files: bool
    ) -> None:
        second = _record()
        lines = [*_dumps(recorded), "", "   ", *_dumps(second), '{"eventTime": "2026-01-01T00:0']
        if per_line_files:
            files = [_write(tmp_path / f"events-{i:05d}.json", [line]) for i, line in enumerate(lines)]
        else:
            files = [_write(tmp_path / "events.ndjson", lines)]

        runs = read_runs(*sorted(files))

        assert len(runs) == 2
        for run, original in zip(runs, (recorded, second), strict=True):
            roots, steps = _split(original)
            assert run == roots + steps

    def test_no_paths_reads_no_runs_and_per_event_files_diff_like_the_appended_file(
        self, recorded: Events, tmp_path: Path
    ) -> None:
        files = [_write(tmp_path / f"events-{i:05d}.json", [line]) for i, line in enumerate(_dumps(recorded))]

        (run,) = read_runs(*sorted(files))

        assert read_runs() == []
        assert diff_run_events(run, recorded).changes == ()

    def test_interleaved_runs_are_still_grouped_per_root(self, recorded: Events, tmp_path: Path) -> None:
        second = _record()
        interleaved = [event for pair in zip(recorded, second, strict=True) for event in pair]
        path = _write(tmp_path / "events.ndjson", _dumps(interleaved))

        runs = read_runs(path)

        assert sorted(len(run) for run in runs) == [len(recorded), len(second)]
        assert {run[0]["run"]["runId"] for run in runs} == {_split(recorded)[0][0]["run"]["runId"]} | {
            _split(second)[0][0]["run"]["runId"]
        }

    def test_a_root_is_found_by_its_facets_not_by_its_job_name(self, tmp_path: Path) -> None:
        events = _record(root_job_name="custom.root")
        path = _write(tmp_path / "events.ndjson", _dumps(events))

        runs = read_runs(path)
        roots, steps = _split(events, "custom.root")

        assert runs == [roots + steps]

    def test_a_root_without_facets_is_found_by_the_steps_naming_it_as_parent(
        self, recorded: Events, tmp_path: Path
    ) -> None:
        bare = _variant(
            recorded,
            lambda e: [_facets(e).pop(name, None) for name in ("mlodaPlan", "processingEngine")],
            job=_ROOT,
        )
        path = _write(tmp_path / "events.ndjson", _dumps(bare))

        roots, steps = _split(bare)

        assert read_runs(path) == [roots + steps]

    def test_a_custom_root_job_name_run_diffs_its_steps(self, tmp_path: Path) -> None:
        before = _record(root_job_name="custom.root")
        after = _variant(
            _record(root_job_name="custom.root"),
            lambda e: e["outputs"][0]["outputFacets"]["outputStatistics"].update(rowCount=99),
            job=_PLUS_ONE,
            kind="COMPLETE",
        )
        path = _write(tmp_path / "events.ndjson", [*_dumps(before), *_dumps(after)])

        first, second = read_runs(path)
        diff = diff_run_events(first, second)

        assert diff.comparable is True
        assert set(_changes(diff.changes)) == {"rowCount"}
