"""Tests for feature_classifications: declared levels, propagation through inputs and the masking step."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from mloda.provider import BaseInputData, FeatureGroup
from mloda.steward import PlanStep
from mloda.user import Options

from mloda.community.extenders.shared.classification import (
    CLASSIFICATION_KEY,
    LEVELS,
    MASKING_ATTRIBUTE,
    feature_classifications,
)


def _declaring(level: str | None) -> Mapping[str, Any]:
    return {} if level is None else {CLASSIFICATION_KEY: level}


class _Undeclared(FeatureGroup):
    pass


class _Public(FeatureGroup):
    @classmethod
    def declared_attributes(cls, features: Any) -> Mapping[str, Any]:
        return _declaring("public")


class _Internal(FeatureGroup):
    @classmethod
    def declared_attributes(cls, features: Any) -> Mapping[str, Any]:
        return _declaring("internal")


class _Pii(FeatureGroup):
    @classmethod
    def declared_attributes(cls, features: Any) -> Mapping[str, Any]:
        return _declaring("pii")


class _Unknown(FeatureGroup):
    @classmethod
    def declared_attributes(cls, features: Any) -> Mapping[str, Any]:
        return _declaring("secret")


class _Masking(FeatureGroup):
    masking = True

    @classmethod
    def declared_attributes(cls, features: Any) -> Mapping[str, Any]:
        return _declaring("internal")


class _MaskingWithoutLevel(FeatureGroup):
    masking = True


class _MaskingAsString(FeatureGroup):
    masking = "true"

    @classmethod
    def declared_attributes(cls, features: Any) -> Mapping[str, Any]:
        return _declaring("internal")


class _PiiReader(BaseInputData):
    @classmethod
    def declared_attributes(cls, features: Any) -> Mapping[str, Any]:
        return _declaring("pii")


class _InternalReader(BaseInputData):
    @classmethod
    def declared_attributes(cls, features: Any) -> Mapping[str, Any]:
        return _declaring("internal")


def _step(
    group: type[FeatureGroup] | None,
    names: tuple[str, ...],
    edges: Mapping[str, tuple[str, ...]] | None = None,
    *,
    reader: type[BaseInputData] | None = None,
    kind: str = "compute",
    options: Options | None = None,
) -> PlanStep:
    return PlanStep(
        step_kind=kind,  # type: ignore[arg-type]
        feature_names=names,
        feature_group=group,
        compute_framework=None,
        source_feature_group=None,
        source_compute_framework=None,
        requested_feature_names=names,
        input_feature_edges=edges or {},
        reader_data_access=None if reader is None else (reader, "access"),
        feature_set_options=options,
    )


def _derived(group: type[FeatureGroup], name: str, source: str) -> PlanStep:
    return _step(group, (name,), {name: (source,)})


def test_the_public_constants() -> None:
    assert CLASSIFICATION_KEY == "mloda.data.classification"
    assert MASKING_ATTRIBUTE == "masking"
    assert LEVELS == ("public", "internal", "confidential", "restricted", "pii")


def test_a_step_without_a_declaration_gets_the_undeclared_level() -> None:
    result = feature_classifications([_step(_Undeclared, ("a",))], undeclared="confidential")

    assert result == {"a": "confidential"}


def test_a_group_declaration_wins_over_undeclared() -> None:
    result = feature_classifications([_step(_Internal, ("a", "b"))], undeclared="pii")

    assert result == {"a": "internal", "b": "internal"}


def test_a_reader_declaration_applies_to_a_group_without_one() -> None:
    result = feature_classifications([_step(_Undeclared, ("a",), reader=_PiiReader)], undeclared="public")

    assert result == {"a": "pii"}


def test_group_and_reader_merge_to_the_most_restrictive_in_either_direction() -> None:
    steps = [_step(_Internal, ("a",), reader=_PiiReader), _step(_Pii, ("b",), reader=_InternalReader)]

    assert feature_classifications(steps, undeclared="public") == {"a": "pii", "b": "pii"}


def test_a_derived_feature_inherits_the_level_of_its_input() -> None:
    steps = [_step(_Pii, ("raw",)), _derived(_Public, "derived", "raw")]

    assert feature_classifications(steps, undeclared="public") == {"raw": "pii", "derived": "pii"}


def test_a_derived_feature_keeps_its_own_higher_level() -> None:
    steps = [_step(_Public, ("raw",)), _derived(_Internal, "derived", "raw")]

    assert feature_classifications(steps, undeclared="public") == {"raw": "public", "derived": "internal"}


def test_the_most_restrictive_of_several_inputs_wins() -> None:
    steps = [
        _step(_Internal, ("a",)),
        _step(_Pii, ("b",)),
        _step(_Public, ("c",), {"c": ("a", "b")}),
    ]

    assert feature_classifications(steps, undeclared="public")["c"] == "pii"


def test_propagation_spans_several_hops() -> None:
    steps = [_step(_Pii, ("a",)), _derived(_Public, "b", "a"), _derived(_Public, "c", "b")]

    assert feature_classifications(steps, undeclared="public")["c"] == "pii"


def test_a_masking_group_lowers_to_its_own_declared_level() -> None:
    steps = [_step(_Pii, ("raw",)), _derived(_Masking, "masked", "raw"), _derived(_Public, "after", "masked")]

    assert feature_classifications(steps, undeclared="public") == {
        "raw": "pii",
        "masked": "internal",
        "after": "internal",
    }


def test_masking_without_its_own_level_lowers_nothing() -> None:
    steps = [_step(_Pii, ("raw",)), _derived(_MaskingWithoutLevel, "masked", "raw")]

    assert feature_classifications(steps, undeclared="public")["masked"] == "pii"


def test_masking_cannot_lower_below_a_reader_only_declaration() -> None:
    class _MaskingReaderOnly(FeatureGroup):
        masking = True

    steps = [_step(_MaskingReaderOnly, ("a",), reader=_PiiReader)]

    assert feature_classifications(steps, undeclared="public") == {"a": "pii"}


def test_a_masking_group_with_a_reader_declares_its_own_level_not_the_readers() -> None:
    steps = [_step(_Masking, ("a",), reader=_PiiReader)]

    assert feature_classifications(steps, undeclared="public") == {"a": "internal"}


def test_a_masking_declared_attribute_without_the_class_attribute_does_not_lower() -> None:
    class _MaskingOnlyDeclared(FeatureGroup):
        @classmethod
        def declared_attributes(cls, features: Any) -> Mapping[str, Any]:
            return {CLASSIFICATION_KEY: "internal", MASKING_ATTRIBUTE: True}

    steps = [_step(_Pii, ("raw",)), _derived(_MaskingOnlyDeclared, "masked", "raw")]

    assert feature_classifications(steps, undeclared="public")["masked"] == "pii"


def test_the_per_feature_masking_option_does_not_lower() -> None:
    options = Options(context={MASKING_ATTRIBUTE: True})
    steps = [_step(_Pii, ("raw",)), _step(_Internal, ("d",), {"d": ("raw",)}, options=options)]

    assert feature_classifications(steps, undeclared="public")["d"] == "pii"


def test_a_masking_value_that_is_not_true_does_not_lower() -> None:
    steps = [_step(_Pii, ("raw",)), _derived(_MaskingAsString, "masked", "raw")]

    assert feature_classifications(steps, undeclared="public")["masked"] == "pii"


def test_several_producers_of_one_name_give_the_most_restrictive() -> None:
    steps = [_step(_Internal, ("a",)), _step(_Pii, ("a",))]

    assert feature_classifications(steps, undeclared="public") == {"a": "pii"}


def test_an_input_without_a_producer_raises_naming_the_feature() -> None:
    with pytest.raises(ValueError, match="ghost"):
        feature_classifications([_derived(_Public, "d", "ghost")], undeclared="public")


def test_an_unknown_declared_level_raises_naming_the_owner() -> None:
    with pytest.raises(ValueError, match="_Unknown|secret"):
        feature_classifications([_step(_Unknown, ("a",))], undeclared="public")


def test_an_invalid_undeclared_level_raises() -> None:
    with pytest.raises(ValueError):
        feature_classifications([_step(_Undeclared, ("a",))], undeclared="secret")


def test_a_cycle_raises() -> None:
    steps = [_derived(_Public, "a", "b"), _derived(_Public, "b", "a")]

    with pytest.raises(ValueError):
        feature_classifications(steps, undeclared="public")


def test_the_result_does_not_depend_on_step_order() -> None:
    steps = [_step(_Pii, ("raw",)), _derived(_Masking, "masked", "raw"), _derived(_Public, "after", "masked")]

    forward = feature_classifications(steps, undeclared="public")

    assert feature_classifications(list(reversed(steps)), undeclared="public") == forward


def test_the_steps_may_be_any_iterable() -> None:
    steps = [_step(_Pii, ("raw",)), _derived(_Public, "d", "raw")]

    assert feature_classifications(iter(steps), undeclared="public")["d"] == "pii"


@pytest.mark.parametrize("kind", ["join", "transform"])
def test_join_and_transform_steps_are_skipped(kind: str) -> None:
    steps = [_step(_Pii, ("a",)), _step(_Pii, ("joined",), kind=kind)]

    assert feature_classifications(steps, undeclared="public") == {"a": "pii"}
