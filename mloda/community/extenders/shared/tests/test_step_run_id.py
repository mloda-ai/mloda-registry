"""Tests for the step identity helpers step_run_id and owner_name."""

from __future__ import annotations

import json
import uuid

import pytest

from mloda.community.extenders.openlineage import openlineage_extender
from mloda.community.extenders.shared import step_run_id as step_run_id_module
from mloda.community.extenders.shared.step_run_id import owner_name, step_run_id
from mloda.testing.extenders.hook_context import make_hook_context

_RUN_ID = "018f1e4a-7c3b-7c3b-8c3b-1234567890ab"
_STEP_UUID = uuid.UUID("6f1c2d3e-4a5b-4c6d-8e7f-0123456789ab")


def test_step_run_id_is_the_uuid5_of_the_run_id_and_the_step_key() -> None:
    expected = str(uuid.uuid5(uuid.UUID(_RUN_ID), json.dumps(["job", ["a", "b"], "PyArrowTable"])))

    assert step_run_id(_RUN_ID, "job", ("b", "a"), "PyArrowTable") == expected


def test_step_run_id_with_a_step_uuid_appends_it_to_the_key() -> None:
    key = json.dumps(["job", ["a", "b"], "PyArrowTable", str(_STEP_UUID)])
    expected = str(uuid.uuid5(uuid.UUID(_RUN_ID), key))

    assert step_run_id(_RUN_ID, "job", ("b", "a"), "PyArrowTable", _STEP_UUID) == expected


def test_step_run_id_differs_only_by_step_uuid() -> None:
    other = uuid.UUID("6f1c2d3e-4a5b-4c6d-8e7f-0123456789ac")

    assert step_run_id(_RUN_ID, "job", ("a",), "fw", _STEP_UUID) != step_run_id(_RUN_ID, "job", ("a",), "fw", other)


def test_step_run_id_without_a_step_uuid_is_the_legacy_value() -> None:
    legacy = str(uuid.uuid5(uuid.UUID(_RUN_ID), json.dumps(["job", ["a"], "fw"])))

    assert step_run_id(_RUN_ID, "job", ("a",), "fw", None) == legacy
    assert step_run_id(_RUN_ID, "job", ("a",), "fw") == legacy


def test_step_run_id_ignores_feature_name_order_and_accepts_any_iterable() -> None:
    first = step_run_id(_RUN_ID, "job", ("a", "b"), "fw")

    assert first == step_run_id(_RUN_ID, "job", ["b", "a"], "fw")
    assert first == step_run_id(_RUN_ID, "job", frozenset({"a", "b"}), "fw")


@pytest.mark.parametrize(
    "other",
    [
        pytest.param({"job_name": "other"}, id="job"),
        pytest.param({"feature_names": ("a",)}, id="features"),
        pytest.param({"compute_framework_name": "Other"}, id="framework"),
        pytest.param({"run_id": "018f1e4a-7c3b-7c3b-8c3b-1234567890ac"}, id="run"),
        pytest.param({"step_uuid": _STEP_UUID}, id="step_uuid"),
    ],
)
def test_step_run_id_differs_when_any_part_of_the_key_differs(other: dict[str, object]) -> None:
    key: dict[str, object] = {
        "run_id": _RUN_ID,
        "job_name": "job",
        "feature_names": ("a", "b"),
        "compute_framework_name": "fw",
    }

    assert step_run_id(**{**key, **other}) != step_run_id(**key)  # type: ignore[arg-type]


def test_step_run_id_is_deterministic_and_a_valid_uuid() -> None:
    value = step_run_id(_RUN_ID, "job", ("a",), None)

    assert value == step_run_id(_RUN_ID, "job", ("a",), None)
    assert value is not None
    assert str(uuid.UUID(value)) == value


@pytest.mark.parametrize("run_id", [None, "", "not-a-uuid", "run-1"])
def test_step_run_id_is_none_without_a_uuid_run_id(run_id: str | None) -> None:
    assert step_run_id(run_id, "job", ("a",), "fw") is None


def test_owner_name_prefers_the_context_feature_group_class() -> None:
    context = make_hook_context(feature_group_class="pkg.mod.Group")

    assert owner_name(context, lambda: None) == "pkg.mod.Group"


def test_owner_name_falls_back_to_the_owning_class_of_the_call() -> None:
    class OwnerFeatureGroup:
        @classmethod
        def calculate_feature(cls) -> None:
            return None

    context = make_hook_context(feature_group_class=None)

    assert owner_name(context, OwnerFeatureGroup.calculate_feature) == "OwnerFeatureGroup"


def test_owner_name_is_still_importable_from_the_openlineage_extender() -> None:
    assert openlineage_extender.owner_name is step_run_id_module.owner_name
