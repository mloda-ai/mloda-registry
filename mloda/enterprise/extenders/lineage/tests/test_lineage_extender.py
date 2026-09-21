"""Tests for LineageFacetsExtender: contract compliance plus the facets it adds to the community emitter. Facet
tests run local feature groups through mloda.run_all; direct __call__ tests use a manually built HookContext."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from collections.abc import Callable, Iterator, Sequence
from typing import Any, ClassVar

import pytest
from mloda.provider import ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.steward import ExtenderHook
from mloda.user import Feature, FeatureName, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from openlineage.client.client import OpenLineageClient
from openlineage.client.event_v2 import InputDataset, OutputDataset, RunEvent, RunState
from openlineage.client.facet_v2 import column_lineage_dataset, data_quality_assertions_dataset, parent_run
from openlineage.client.serde import Serde

from mloda.enterprise.extenders.lineage.lineage_extender import LineageFacetsExtender, MlodaRunFacet
from mloda.testing.extenders.hook_context import make_hook_context
from mloda.testing.extenders.openlineage import OpenLineageExtenderTestMixin, RecordingTransport, make_recording_client

_LINEAGE_PRODUCER = "https://github.com/mloda-ai/mloda-registry/tree/main/mloda/enterprise/extenders/lineage"
_CUSTOM_PRODUCER = "https://example.invalid/custom-lineage-producer"
_STEP_LEVEL_MARKER = "step-level declared inputs"
_ROW_VALUE_MARKER = "SENSITIVE_ROW_VALUE_xyz123"

_VALIDATE_INPUT = "validate_input_features"
_VALIDATE_OUTPUT = "validate_output_features"
_VALIDATION_JOB_SUFFIXES = (f".{_VALIDATE_INPUT}", f".{_VALIDATE_OUTPUT}")

_ROOT = "lineage_facets_root"
_ROOT_A = "lineage_facets_root_a"
_ROOT_B = "lineage_facets_root_b"


class _Root(FeatureGroup):
    """Root step: declares no inputs, so it never carries column lineage."""

    @classmethod
    def input_data(cls) -> DataCreator:
        return DataCreator({_ROOT, _ROOT_A, _ROOT_B})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {name: [1, 2, 3] for name in features.get_all_names()}


class _Derived(FeatureGroup):
    """Derived step with the default validators: one column per requested output, computed from `inputs`."""

    outputs: ClassVar[tuple[str, ...]] = ("lineage_facets_derived",)
    inputs: ClassVar[tuple[str, ...]] = (_ROOT,)

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return set(cls.outputs)

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(name) for name in self.inputs}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {name: [1, 2, 3] for name in features.get_all_names()}


class _TwoInputs(_Derived):
    outputs = ("lineage_facets_two_inputs",)
    inputs = (_ROOT_B, _ROOT_A)


class _MultiOutput(_Derived):
    outputs = tuple(f"lineage_facets_multi_{suffix}" for suffix in "fcaebd")


class _MaskedByAttribute(_Derived):
    outputs = ("lineage_facets_masked_attribute",)
    masking = True


class _MaskedByStringAttribute(_Derived):
    outputs = ("lineage_facets_masked_string_attribute",)
    masking = "true"


class _PassingValidators(_Derived):
    outputs = ("lineage_facets_passing_a", "lineage_facets_passing_b")
    inputs = (_ROOT_B, _ROOT_A)

    @classmethod
    def validate_input_features(cls, data: Any, features: FeatureSet) -> None:
        return None

    @classmethod
    def validate_output_features(cls, data: Any, features: FeatureSet) -> None:
        return None


class _InputValidatorOnly(_Derived):
    outputs = ("lineage_facets_input_only",)

    @classmethod
    def validate_input_features(cls, data: Any, features: FeatureSet) -> None:
        return None


class _FailingInputValidator(_Derived):
    outputs = ("lineage_facets_failing_input",)

    @classmethod
    def validate_input_features(cls, data: Any, features: FeatureSet) -> None:
        raise ValueError(f"invalid value found: {_ROW_VALUE_MARKER}")


class _FailingOutputValidator(_Derived):
    outputs = ("lineage_facets_failing_output",)

    @classmethod
    def validate_output_features(cls, data: Any, features: FeatureSet) -> None:
        raise ValueError(f"invalid value found: {_ROW_VALUE_MARKER}")


class _Interrupt(BaseException):
    pass


class _InterruptedValidator(_Derived):
    outputs = ("lineage_facets_interrupted",)

    @classmethod
    def validate_output_features(cls, data: Any, features: FeatureSet) -> None:
        raise _Interrupt(f"interrupted: {_ROW_VALUE_MARKER}")


class _CustomProducerLineageExtender(LineageFacetsExtender):
    producer = _CUSTOM_PRODUCER


def _job(feature_group: type[FeatureGroup], suffix: str = "") -> str:
    return f"{feature_group.__module__}.{feature_group.__qualname__}{suffix}"


def _calculate_run_events(events: list[RunEvent]) -> list[RunEvent]:
    return [event for event in events if not event.job.name.endswith(_VALIDATION_JOB_SUFFIXES)]


def _run(
    extender: LineageFacetsExtender, features: Sequence[Feature | str], *feature_groups: type[FeatureGroup]
) -> None:
    mloda.run_all(
        list(features),
        compute_frameworks={PyArrowTable},
        plugin_collector=PluginCollector.enabled_feature_groups({_Root, *feature_groups}),
        function_extender={extender},
    )


def _events_for(events: list[RunEvent], job: str) -> list[RunEvent]:
    return [event for event in events if event.job.name == job]


def _complete(events: list[RunEvent], job: str) -> RunEvent:
    (event,) = [e for e in _events_for(events, job) if e.eventType == RunState.COMPLETE]
    return event


def _parent(event: RunEvent) -> parent_run.ParentRunFacet:
    parent = (event.run.facets or {}).get("parent")
    assert isinstance(parent, parent_run.ParentRunFacet)
    return parent


def _parent_run_id(event: RunEvent) -> str:
    return _parent(event).run.runId


def _root_run_id(events: list[RunEvent]) -> str:
    return _parent_run_id(_events_for(events, _job(_Root))[0])


def _run_facet(event: RunEvent) -> MlodaRunFacet:
    facet = (event.run.facets or {}).get("mloda")
    assert isinstance(facet, MlodaRunFacet)
    return facet


def _output(event: RunEvent, name: str) -> OutputDataset:
    (dataset,) = [output for output in event.outputs or [] if output.name == name]
    return dataset


def _column_lineage(event: RunEvent, name: str) -> column_lineage_dataset.ColumnLineageDatasetFacet:
    facet = (_output(event, name).facets or {}).get("columnLineage")
    assert isinstance(facet, column_lineage_dataset.ColumnLineageDatasetFacet)
    return facet


def _transformations(event: RunEvent, name: str) -> list[column_lineage_dataset.Transformation]:
    fields = _column_lineage(event, name).fields[name]
    return [t for input_field in fields.inputFields for t in input_field.transformations or []]


def _assertions(dataset: InputDataset) -> list[tuple[str, bool]]:
    facet = (dataset.inputFacets or {}).get("dataQualityAssertions")
    assert isinstance(facet, data_quality_assertions_dataset.DataQualityAssertionsDatasetFacet)
    return [(assertion.assertion, assertion.success) for assertion in facet.assertions]


def _own_facet_producers(event: RunEvent) -> list[tuple[str, str]]:
    """(facet key, _producer) of every facet the extender builds, not client-injected run facets."""
    payload = json.loads(Serde.to_json(event))
    run_facets = (payload.get("run") or {}).get("facets") or {}
    found = [(key, run_facets[key]["_producer"]) for key in ("parent", "mloda") if key in run_facets]
    for dataset in (payload.get("inputs") or []) + (payload.get("outputs") or []):
        for group in ("facets", "inputFacets"):
            found.extend((key, facet["_producer"]) for key, facet in (dataset.get(group) or {}).items())
    return found


def _expected_structure_hash(
    *,
    feature_group_class: str,
    feature_group_version: str,
    plugin_version: str | None,
    compute_framework: str,
    feature_names: list[str],
    input_features: list[str],
    masked_features: list[str],
) -> str:
    structure = [
        feature_group_class,
        feature_group_version,
        plugin_version,
        compute_framework,
        feature_names,
        input_features,
        masked_features,
    ]
    return hashlib.sha256(json.dumps(structure, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _structure_hash(func: Callable[..., Any] | None = None, args: tuple[Any, ...] = (), **context: Any) -> str:
    client, transport = make_recording_client()
    with make_hook_context(**context).activate():
        LineageFacetsExtender(client=client)(func or (lambda: None), *args)
    return _run_facet(transport.events[0]).structureHash


@pytest.fixture
def ol_capture() -> Iterator[tuple[OpenLineageClient, RecordingTransport]]:
    """A fresh, isolated (client, transport) pair per test."""
    yield make_recording_client()


class TestLineageFacetsExtenderContract(OpenLineageExtenderTestMixin):
    """LineageFacetsExtender must satisfy the shared Extender contract and the OpenLineage RunEvent contract."""

    @classmethod
    def extender_class(cls) -> type[LineageFacetsExtender]:
        return LineageFacetsExtender

    def make_openlineage_extender(
        self, client: OpenLineageClient, *, raise_on_error: bool | None = None
    ) -> LineageFacetsExtender:
        if raise_on_error is None:
            return LineageFacetsExtender(client=client)
        return LineageFacetsExtender(client=client, raise_on_error=raise_on_error)

    @classmethod
    def expected_hooks(cls) -> set[ExtenderHook] | None:
        return {
            ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
            ExtenderHook.INPUT_DATA_LOAD,
            ExtenderHook.VALIDATE_INPUT_FEATURE,
            ExtenderHook.VALIDATE_OUTPUT_FEATURE,
        }

    @classmethod
    def emits_schema_facets(cls) -> bool:
        return True

    def calculate_run_events(self, events: list[RunEvent]) -> list[RunEvent]:
        return _calculate_run_events(events)


class TestLineageFacetsColumnLineage:
    """One columnLineage facet per COMPLETE output of a step that declares input features."""

    def test_derived_output_carries_direct_lineage_from_its_declared_input(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        extender = LineageFacetsExtender(client=client, dataset_namespace="lineage-ds")

        _run(extender, ["lineage_facets_derived"], _Derived)

        complete = _complete(transport.events, _job(_Derived))
        expected = column_lineage_dataset.Fields(
            inputFields=[
                column_lineage_dataset.InputField(
                    namespace="lineage-ds",
                    name=_ROOT,
                    field=_ROOT,
                    transformations=[column_lineage_dataset.Transformation(type="DIRECT")],
                )
            ]
        )
        assert _column_lineage(complete, "lineage_facets_derived").fields == {"lineage_facets_derived": expected}
        assert {"schema", "columnLineage"} <= set(_output(complete, "lineage_facets_derived").facets or {})

    def test_input_fields_are_sorted_by_name(self, ol_capture: tuple[OpenLineageClient, RecordingTransport]) -> None:
        client, transport = ol_capture
        names = {f"input_{index}" for index in (5, 2, 7, 1, 9, 3, 8, 0)}

        with make_hook_context(feature_names=("out",), input_features=frozenset(names)).activate():
            LineageFacetsExtender(client=client)(lambda: None)

        input_fields = _column_lineage(transport.events[-1], "out").fields["out"].inputFields
        assert [input_field.name for input_field in input_fields] == sorted(names)
        assert [input_field.field for input_field in input_fields] == sorted(names)

    def test_run_all_lists_the_declared_inputs_sorted(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture

        _run(LineageFacetsExtender(client=client), list(_TwoInputs.outputs), _TwoInputs)

        complete = _complete(transport.events, _job(_TwoInputs))
        input_fields = _column_lineage(complete, _TwoInputs.outputs[0]).fields[_TwoInputs.outputs[0]].inputFields
        assert [input_field.name for input_field in input_fields] == [_ROOT_A, _ROOT_B]

    def test_root_step_has_no_column_lineage(self, ol_capture: tuple[OpenLineageClient, RecordingTransport]) -> None:
        client, transport = ol_capture

        _run(LineageFacetsExtender(client=client), ["lineage_facets_derived"], _Derived)

        outputs = _complete(transport.events, _job(_Root)).outputs or []
        assert [output.name for output in outputs] == [_ROOT]
        assert "schema" in (outputs[0].facets or {})
        assert "columnLineage" not in (outputs[0].facets or {})

    @pytest.mark.parametrize("input_features", [None, frozenset[str]()])
    def test_no_column_lineage_without_declared_inputs(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], input_features: frozenset[str] | None
    ) -> None:
        client, transport = ol_capture

        with make_hook_context(feature_names=("out",), input_features=input_features).activate():
            LineageFacetsExtender(client=client)(lambda: None)

        assert "columnLineage" not in (_output(transport.events[-1], "out").facets or {})

    @pytest.mark.parametrize(
        ("feature_names", "description"),
        [(("one",), None), (("one", "two"), _STEP_LEVEL_MARKER)],
        ids=["single output", "multi output"],
    )
    def test_step_level_marker_is_set_only_on_multi_output_steps(
        self,
        ol_capture: tuple[OpenLineageClient, RecordingTransport],
        feature_names: tuple[str, ...],
        description: str | None,
    ) -> None:
        client, transport = ol_capture

        with make_hook_context(feature_names=feature_names, input_features=frozenset({"src"})).activate():
            LineageFacetsExtender(client=client)(lambda: None)

        for name in feature_names:
            assert _transformations(transport.events[-1], name) == [
                column_lineage_dataset.Transformation(type="DIRECT", description=description)
            ]

    def test_run_all_multi_output_step_marks_every_output_edge_as_step_level(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture

        _run(LineageFacetsExtender(client=client), list(_MultiOutput.outputs), _MultiOutput)

        complete = _complete(transport.events, _job(_MultiOutput))
        assert sorted(output.name for output in complete.outputs or []) == sorted(_MultiOutput.outputs)
        for name in _MultiOutput.outputs:
            assert set(_column_lineage(complete, name).fields) == {name}
            assert _transformations(complete, name) == [
                column_lineage_dataset.Transformation(type="DIRECT", description=_STEP_LEVEL_MARKER)
            ]


_MASKED_CASES = [
    pytest.param(_MaskedByAttribute, lambda: Options(), id="class attribute"),
    pytest.param(_Derived, lambda: Options(context={"masking": True}), id="own context option"),
]

_NOT_MASKED_CASES = [
    pytest.param(_Derived, lambda: Options(), id="undeclared"),
    pytest.param(_Derived, lambda: Options(group={"masking": True}), id="group key"),
    pytest.param(_Derived, lambda: Options(context={"masking": "true"}), id="context string"),
    pytest.param(_Derived, lambda: Options(context={"masking": False}), id="context false"),
    pytest.param(_MaskedByStringAttribute, lambda: Options(), id="class attribute string"),
]


class TestLineageFacetsMasking:
    """Masking is declared, never inferred: a class attribute or the feature's own context option, both `is True`."""

    @pytest.mark.parametrize(("feature_group", "make_options"), _MASKED_CASES)
    def test_declared_masking_marks_the_transformation_and_lists_the_feature(
        self,
        ol_capture: tuple[OpenLineageClient, RecordingTransport],
        feature_group: type[_Derived],
        make_options: Callable[[], Options],
    ) -> None:
        client, transport = ol_capture
        name = feature_group.outputs[0]

        _run(LineageFacetsExtender(client=client), [Feature(name, options=make_options())], feature_group)

        events = _events_for(transport.events, _job(feature_group))
        assert [event.eventType for event in events] == [RunState.START, RunState.COMPLETE]
        assert [t.masking for t in _transformations(events[-1], name)] == [True]
        assert all(_run_facet(event).maskedFeatures == [name] for event in events)
        assert all(_run_facet(event).maskedFeatures == [] for event in _events_for(transport.events, _job(_Root)))

    @pytest.mark.parametrize(("feature_group", "make_options"), _NOT_MASKED_CASES)
    def test_undeclared_or_misdeclared_masking_is_ignored(
        self,
        ol_capture: tuple[OpenLineageClient, RecordingTransport],
        feature_group: type[_Derived],
        make_options: Callable[[], Options],
    ) -> None:
        client, transport = ol_capture
        name = feature_group.outputs[0]

        _run(LineageFacetsExtender(client=client), [Feature(name, options=make_options())], feature_group)

        events = _events_for(transport.events, _job(feature_group))
        assert [event.eventType for event in events] == [RunState.START, RunState.COMPLETE]
        assert [t.masking for t in _transformations(events[-1], name)] == [None]
        assert all(_run_facet(event).maskedFeatures == [] for event in events)
        assert all(_run_facet(event).maskedFeatures == [] for event in _events_for(transport.events, _job(_Root)))

    def test_a_forwarded_context_key_masks_only_the_declaring_step(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        """propagate_context_keys copies the key onto the input features, so the upstream step's own options carry
        it too; core marks it inherited and only the step that declared it reports masking."""
        client, transport = ol_capture
        name = _Derived.outputs[0]
        options = Options(context={"masking": True}, propagate_context_keys=frozenset({"masking"}))

        _run(LineageFacetsExtender(client=client), [Feature(name, options=options)], _Derived)

        derived_events = _events_for(transport.events, _job(_Derived))
        assert [t.masking for t in _transformations(derived_events[-1], name)] == [True]
        assert all(_run_facet(event).maskedFeatures == [name] for event in derived_events)
        assert all(_run_facet(event).maskedFeatures == [] for event in _events_for(transport.events, _job(_Root)))

    def test_masking_is_declared_per_feature_within_a_step(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        plain = "lineage_facets_multi_b"
        masked = [name for name in _MultiOutput.outputs if name != plain]
        features: list[Feature | str] = [Feature(name, options=Options(context={"masking": True})) for name in masked]
        features.append(Feature(plain))

        _run(LineageFacetsExtender(client=client), features, _MultiOutput)

        events = _events_for(transport.events, _job(_MultiOutput))
        assert [event.eventType for event in events] == [RunState.START, RunState.COMPLETE]
        assert all(_run_facet(event).maskedFeatures == sorted(masked) for event in events)
        for name in masked:
            assert [t.masking for t in _transformations(events[-1], name)] == [True]
        assert [t.masking for t in _transformations(events[-1], plain)] == [None]

    def test_class_attribute_masks_every_feature_of_the_step_sorted(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture

        with make_hook_context(feature_names=("zeta", "alpha"), input_features=frozenset({"src"})).activate():
            LineageFacetsExtender(client=client)(_MaskedByAttribute.calculate_feature, None, FeatureSet())

        assert all(_run_facet(event).maskedFeatures == ["alpha", "zeta"] for event in transport.events)
        for name in ("alpha", "zeta"):
            assert [t.masking for t in _transformations(transport.events[-1], name)] == [True]


class TestLineageFacetsValidationRuns:
    """An overridden validator is its own nested Run carrying a dataQualityAssertions facet; a default no-op emits none."""

    def test_calculate_run_events_drops_the_validation_runs(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture

        _run(LineageFacetsExtender(client=client), list(_PassingValidators.outputs), _PassingValidators)

        assert {event.job.name for event in transport.events} == {
            _job(_Root),
            _job(_PassingValidators),
            _job(_PassingValidators, f".{_VALIDATE_INPUT}"),
            _job(_PassingValidators, f".{_VALIDATE_OUTPUT}"),
        }
        assert {event.job.name for event in _calculate_run_events(transport.events)} == {
            _job(_Root),
            _job(_PassingValidators),
        }

    def test_default_validators_emit_no_validation_run(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture

        _run(LineageFacetsExtender(client=client), ["lineage_facets_derived"], _Derived)

        assert [event.job.name for event in transport.events] == [_job(_Root)] * 2 + [_job(_Derived)] * 2

    def test_only_an_overridden_validator_emits_a_run(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture

        _run(LineageFacetsExtender(client=client), list(_InputValidatorOnly.outputs), _InputValidatorOnly)

        validation_jobs = {
            event.job.name for event in transport.events if event.job.name.endswith(_VALIDATION_JOB_SUFFIXES)
        }
        assert validation_jobs == {_job(_InputValidatorOnly, f".{_VALIDATE_INPUT}")}

    @pytest.mark.parametrize("method", [_VALIDATE_INPUT, _VALIDATE_OUTPUT])
    def test_passing_validator_emits_a_nested_run_with_a_passing_assertion_per_feature(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], method: str
    ) -> None:
        client, transport = ol_capture
        extender = LineageFacetsExtender(client=client, job_namespace="lineage-jobs", dataset_namespace="lineage-ds")
        validated = {_VALIDATE_INPUT: [_ROOT_A, _ROOT_B], _VALIDATE_OUTPUT: sorted(_PassingValidators.outputs)}

        _run(extender, list(_PassingValidators.outputs), _PassingValidators)

        run_events = _events_for(transport.events, _job(_PassingValidators, f".{method}"))
        assert [event.eventType for event in run_events] == [RunState.START, RunState.COMPLETE]
        start, complete = run_events
        root_run_id = _root_run_id(transport.events)
        calculate_run_id = _events_for(transport.events, _job(_PassingValidators))[0].run.runId
        assert start.run.runId == complete.run.runId
        assert start.run.runId not in {root_run_id, calculate_run_id}
        for event in run_events:
            assert event.job.namespace == "lineage-jobs"
            assert _parent_run_id(event) == root_run_id
            assert (_parent(event).job.namespace, _parent(event).job.name) == ("lineage-jobs", "mloda.run_all")
        assert start.inputs == []
        datasets = complete.inputs or []
        assert sorted((dataset.namespace, dataset.name) for dataset in datasets) == [
            ("lineage-ds", name) for name in validated[method]
        ]
        assert all(_assertions(dataset) == [(method, True)] for dataset in datasets)

    def test_input_validation_falls_back_to_feature_names_without_declared_inputs(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        context = make_hook_context(
            hook=ExtenderHook.VALIDATE_INPUT_FEATURE,
            feature_group_class=_job(_PassingValidators),
            feature_names=("y", "x"),
            input_features=None,
        )

        with context.activate():
            LineageFacetsExtender(client=client)(_PassingValidators.validate_input_features, None, FeatureSet())

        complete = transport.events[-1]
        assert complete.eventType == RunState.COMPLETE
        assert sorted(dataset.name for dataset in complete.inputs or []) == ["x", "y"]
        assert all(_assertions(dataset) == [(_VALIDATE_INPUT, True)] for dataset in complete.inputs or [])

    @pytest.mark.parametrize(
        ("feature_group", "method"),
        [(_FailingInputValidator, _VALIDATE_INPUT), (_FailingOutputValidator, _VALIDATE_OUTPUT)],
        ids=["input", "output"],
    )
    def test_failing_validator_ends_in_fail_reraises_and_never_leaks_the_message(
        self,
        ol_capture: tuple[OpenLineageClient, RecordingTransport],
        feature_group: type[_Derived],
        method: str,
    ) -> None:
        client, transport = ol_capture

        with pytest.raises(ValueError, match=_ROW_VALUE_MARKER):
            _run(LineageFacetsExtender(client=client), list(feature_group.outputs), feature_group)

        run_events = _events_for(transport.events, _job(feature_group, f".{method}"))
        assert [event.eventType for event in run_events] == [RunState.START, RunState.FAIL]
        assert all(_parent_run_id(event) == _root_run_id(transport.events) for event in run_events)
        datasets = run_events[-1].inputs or []
        assert datasets
        assert all(_assertions(dataset) == [(method, False)] for dataset in datasets)
        for event in transport.events:
            assert _ROW_VALUE_MARKER not in Serde.to_json(event)

    def test_interrupted_validator_ends_in_abort_and_propagates(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        context = make_hook_context(
            hook=ExtenderHook.VALIDATE_OUTPUT_FEATURE,
            feature_group_class=_job(_InterruptedValidator),
            feature_names=("out",),
            run_id=str(uuid.uuid4()),
        )

        with context.activate():
            with pytest.raises(_Interrupt):
                LineageFacetsExtender(client=client)(_InterruptedValidator.validate_output_features, None, FeatureSet())

        assert [event.eventType for event in transport.events] == [RunState.START, RunState.ABORT]
        datasets = transport.events[-1].inputs or []
        assert [dataset.name for dataset in datasets] == ["out"]
        assert _assertions(datasets[0]) == [(_VALIDATE_OUTPUT, False)]
        for event in transport.events:
            assert _ROW_VALUE_MARKER not in Serde.to_json(event)

    @pytest.mark.parametrize("hook", [ExtenderHook.VALIDATE_INPUT_FEATURE, ExtenderHook.VALIDATE_OUTPUT_FEATURE])
    def test_bare_validator_hook_call_passes_through_without_events(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], hook: ExtenderHook
    ) -> None:
        client, transport = ol_capture

        with make_hook_context(hook=hook).activate():
            assert LineageFacetsExtender(client=client)(lambda: 42) == 42

        assert transport.events == []


class TestLineageFacetsRunFacet:
    """The custom `mloda` run facet: context values plus a deterministic structure hash."""

    def test_present_on_start_and_terminal_events_with_the_context_values(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture
        run_id = str(uuid.uuid4())
        context = make_hook_context(
            run_id=run_id,
            feature_group_version="7",
            plugin_version="2.5.0",
            compute_framework_name="MyFramework",
            feature_names=("beta", "alpha"),
            input_features=frozenset({"src"}),
        )

        with context.activate():
            LineageFacetsExtender(client=client)(lambda: None)

        assert [event.eventType for event in transport.events] == [RunState.START, RunState.COMPLETE]
        for event in transport.events:
            facet = _run_facet(event)
            assert (facet.featureGroupVersion, facet.pluginVersion, facet.computeFramework) == (
                "7",
                "2.5.0",
                "MyFramework",
            )
            assert facet.maskedFeatures == []
            assert _parent_run_id(event) == run_id
            payload = json.loads(Serde.to_json(event))
            assert payload["run"]["facets"]["mloda"]["structureHash"] == facet.structureHash
        assert _run_facet(transport.events[0]).structureHash == _run_facet(transport.events[1]).structureHash

    def test_present_on_the_fail_event(self, ol_capture: tuple[OpenLineageClient, RecordingTransport]) -> None:
        client, transport = ol_capture

        def failing() -> None:
            raise RuntimeError("calculate boom")

        with make_hook_context().activate():
            with pytest.raises(RuntimeError, match="calculate boom"):
                LineageFacetsExtender(client=client)(failing)

        assert [event.eventType for event in transport.events] == [RunState.START, RunState.FAIL]
        assert _run_facet(transport.events[0]).structureHash == _run_facet(transport.events[1]).structureHash

    def test_plugin_version_none_is_passed_through(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport]
    ) -> None:
        client, transport = ol_capture

        with make_hook_context(plugin_version=None).activate():
            LineageFacetsExtender(client=client)(lambda: None)

        assert all(_run_facet(event).pluginVersion is None for event in transport.events)

    def test_schema_url_is_a_stable_https_url_of_the_module(self) -> None:
        url = MlodaRunFacet._get_schema()

        assert url.startswith("https://")
        assert "mloda/enterprise/extenders/lineage" in url
        assert url != "https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/RunFacet"

    def test_structure_hash_is_the_sha256_of_the_canonical_structure(self) -> None:
        actual = _structure_hash(
            feature_group_class="pkg.Fg",
            feature_group_version="v3",
            plugin_version="1.2.3",
            compute_framework_name="PyArrowTable",
            feature_names=("b", "a"),
            input_features=frozenset({"y", "x", "z"}),
        )

        assert re.fullmatch(r"[0-9a-f]{64}", actual)
        assert actual == _expected_structure_hash(
            feature_group_class="pkg.Fg",
            feature_group_version="v3",
            plugin_version="1.2.3",
            compute_framework="PyArrowTable",
            feature_names=["a", "b"],
            input_features=["x", "y", "z"],
            masked_features=[],
        )

    def test_structure_hash_includes_the_declared_masking(self) -> None:
        actual = _structure_hash(
            _MaskedByAttribute.calculate_feature,
            (None, FeatureSet()),
            feature_group_class="pkg.Fg",
            feature_group_version="1",
            plugin_version=None,
            compute_framework_name="PyArrowTable",
            feature_names=("b", "a"),
            input_features=frozenset({"src"}),
        )

        assert actual == _expected_structure_hash(
            feature_group_class="pkg.Fg",
            feature_group_version="1",
            plugin_version=None,
            compute_framework="PyArrowTable",
            feature_names=["a", "b"],
            input_features=["src"],
            masked_features=["a", "b"],
        )

    def test_structure_hash_ignores_run_identity(self) -> None:
        first = _structure_hash(run_id=str(uuid.uuid4()), tenant_id="tenant-1", principal="alice", worker_index=0)
        second = _structure_hash(run_id=str(uuid.uuid4()), tenant_id="tenant-2", principal="bob", worker_index=3)

        assert first == second

    def test_structure_hash_is_stable_across_run_all_runs(self) -> None:
        def start_hashes() -> dict[str, str]:
            client, transport = make_recording_client()
            _run(LineageFacetsExtender(client=client), ["lineage_facets_derived"], _Derived)
            return {
                event.job.name: _run_facet(event).structureHash
                for event in transport.events
                if event.eventType == RunState.START
            }

        first, second = start_hashes(), start_hashes()

        assert set(first) == {_job(_Root), _job(_Derived)}
        assert first == second
        assert first[_job(_Root)] != first[_job(_Derived)]

    @pytest.mark.parametrize(
        ("field", "changed"),
        [
            ("feature_group_class", "pkg.Other"),
            ("feature_group_version", "2"),
            ("plugin_version", "1.0.1"),
            ("compute_framework_name", "PandasDataFrame"),
            ("feature_names", ("out", "extra")),
            ("input_features", frozenset({"src", "extra_src"})),
        ],
    )
    def test_structure_hash_differs_when_the_structure_differs(self, field: str, changed: Any) -> None:
        base: dict[str, Any] = {
            "feature_group_class": "pkg.Fg",
            "feature_group_version": "1",
            "plugin_version": "1.0.0",
            "compute_framework_name": "PyArrowTable",
            "feature_names": ("out",),
            "input_features": frozenset({"src"}),
        }

        assert _structure_hash(**base) != _structure_hash(**{**base, field: changed})

    def test_structure_hash_differs_when_a_feature_is_masked(self) -> None:
        context: dict[str, Any] = {"feature_names": ("out",), "input_features": frozenset({"src"})}

        unmasked = _structure_hash(**context)
        masked = _structure_hash(_MaskedByAttribute.calculate_feature, (None, FeatureSet()), **context)

        assert unmasked != masked


class TestLineageFacetsProducer:
    """Every event and every facet the class builds carries the extender's own producer."""

    @pytest.mark.parametrize(
        ("extender_class", "producer"),
        [(LineageFacetsExtender, _LINEAGE_PRODUCER), (_CustomProducerLineageExtender, _CUSTOM_PRODUCER)],
        ids=["default", "custom"],
    )
    def test_every_event_and_facet_uses_the_extender_producer(
        self,
        ol_capture: tuple[OpenLineageClient, RecordingTransport],
        extender_class: type[LineageFacetsExtender],
        producer: str,
    ) -> None:
        client, transport = ol_capture

        _run(extender_class(client=client), list(_PassingValidators.outputs), _PassingValidators)

        assert extender_class.producer == producer
        assert transport.events
        assert all(event.producer == producer for event in transport.events)
        facets = [facet for event in transport.events for facet in _own_facet_producers(event)]
        assert {key for key, _ in facets} >= {"parent", "mloda", "schema", "columnLineage", "dataQualityAssertions"}
        assert {facet_producer for _, facet_producer in facets} == {producer}


class TestLineageFacetsBareCalls:
    """A bare callable under a HookContext without a FeatureSet must never break a facet builder."""

    def test_bare_calculate_call_builds_every_facet_without_a_feature_set(
        self, ol_capture: tuple[OpenLineageClient, RecordingTransport], caplog: pytest.LogCaptureFixture
    ) -> None:
        client, transport = ol_capture

        with caplog.at_level(logging.WARNING):
            with make_hook_context(feature_names=("out",), input_features=frozenset({"src"})).activate():
                assert LineageFacetsExtender(client=client)(lambda: None) is None

        assert [event.eventType for event in transport.events] == [RunState.START, RunState.COMPLETE]
        assert all(_run_facet(event).maskedFeatures == [] for event in transport.events)
        assert [t.masking for t in _transformations(transport.events[-1], "out")] == [None]
        assert [record for record in caplog.records if record.levelno >= logging.WARNING] == []
