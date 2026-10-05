"""Tests for step_run_id and owner_name: the step identity helpers shared by the OpenLineage, OTel and audit extenders."""

from __future__ import annotations

import json
import uuid

import pytest

from mloda.community.extenders.openlineage import openlineage_extender
from mloda.community.extenders.shared import step_run_id as step_run_id_module
from mloda.community.extenders.shared.step_run_id import owner_name, step_run_id
from mloda.testing.extenders.hook_context import make_hook_context

_RUN_ID = "018f1e4a-7c3b-7c3b-8c3b-1234567890ab"


def test_step_run_id_is_the_uuid5_of_the_run_id_and_the_step_key() -> None:
    expected = str(uuid.uuid5(uuid.UUID(_RUN_ID), json.dumps(["job", ["a", "b"], "PyArrowTable"])))

    assert step_run_id(_RUN_ID, "job", ("b", "a"), "PyArrowTable") == expected


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
