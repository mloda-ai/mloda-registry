"""Tests for feature_classifications: declared levels, propagation through inputs and the masking step."""

from __future__ import annotations

import logging
import pickle  # nosec
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
    RunClassifications,
    feature_classifications,
)
from mloda.community.extenders.shared.tests.classification_helpers import (
    InternalGroup,
    PiiGroup,
    classified_step,
    declaring,
)


class _Undeclared(FeatureGroup):
    pass


class _Public(FeatureGroup):
    @classmethod
    def declared_attributes(cls, features: Any) -> Mapping[str, Any]:
        return declaring("public")


class _Unknown(FeatureGroup):
    @classmethod
    def declared_attributes(cls, features: Any) -> Mapping[str, Any]:
        return declaring("secret")


class _Masking(FeatureGroup):
    masking = True

    @classmethod
    def declared_attributes(cls, features: Any) -> Mapping[str, Any]:
        return declaring("internal")


class _MaskingWithoutLevel(FeatureGroup):
    masking = True


class _MaskingAsString(FeatureGroup):
    masking = "true"

    @classmethod
    def declared_attributes(cls, features: Any) -> Mapping[str, Any]:
        return declaring("internal")


class _PiiReader(BaseInputData):
    @classmethod
    def declared_attributes(cls, features: Any) -> Mapping[str, Any]:
        return declaring("pii")


class _InternalReader(BaseInputData):
    @classmethod
    def declared_attributes(cls, features: Any) -> Mapping[str, Any]:
        return declaring("internal")


class _PublicReader(BaseInputData):
    @classmethod
    def declared_attributes(cls, features: Any) -> Mapping[str, Any]:
        return declaring("public")


class _UndeclaredReader(BaseInputData):
    pass


def _derived(group: type[FeatureGroup], name: str, source: str) -> PlanStep:
    return classified_step(group, (name,), {name: (source,)})


def test_the_public_constants() -> None:
    assert CLASSIFICATION_KEY == "mloda.data.classification"
    assert MASKING_ATTRIBUTE == "masking"
    assert LEVELS == ("public", "internal", "confidential", "restricted", "pii")


def test_a_step_without_a_declaration_gets_the_undeclared_level() -> None:
    result = feature_classifications([classified_step(_Undeclared, ("a",))], undeclared="confidential")

    assert result == {"a": "confidential"}


def test_a_group_declaration_wins_over_undeclared() -> None:
    result = feature_classifications([classified_step(InternalGroup, ("a", "b"))], undeclared="pii")

    assert result == {"a": "internal", "b": "internal"}


def test_a_reader_declaration_applies_to_a_group_without_one() -> None:
    result = feature_classifications([classified_step(_Undeclared, ("a",), reader=_PiiReader)], undeclared="public")

    assert result == {"a": "pii"}


def test_group_and_reader_merge_to_the_most_restrictive_in_either_direction() -> None:
    steps = [
        classified_step(InternalGroup, ("a",), reader=_PiiReader),
        classified_step(PiiGroup, ("b",), reader=_InternalReader),
    ]

    assert feature_classifications(steps, undeclared="public") == {"a": "pii", "b": "pii"}


def test_a_derived_feature_inherits_the_level_of_its_input() -> None:
    steps = [classified_step(PiiGroup, ("raw",)), _derived(_Public, "derived", "raw")]

    assert feature_classifications(steps, undeclared="public") == {"raw": "pii", "derived": "pii"}


def test_a_derived_feature_keeps_its_own_higher_level() -> None:
    steps = [classified_step(_Public, ("raw",)), _derived(InternalGroup, "derived", "raw")]

    assert feature_classifications(steps, undeclared="public") == {"raw": "public", "derived": "internal"}


@pytest.mark.parametrize("group, expected", [(_Public, "pii"), (_Masking, "internal")])
def test_the_most_restrictive_of_several_inputs_wins(group: type[FeatureGroup], expected: str) -> None:
    steps = [
        classified_step(InternalGroup, ("a",)),
        classified_step(PiiGroup, ("b",)),
        classified_step(group, ("c",), {"c": ("a", "b")}),
    ]

    assert feature_classifications(steps, undeclared="public")["c"] == expected


def test_propagation_spans_several_hops() -> None:
    steps = [classified_step(PiiGroup, ("a",)), _derived(_Public, "b", "a"), _derived(_Public, "c", "b")]

    assert feature_classifications(steps, undeclared="public")["c"] == "pii"


def test_a_masking_group_lowers_to_its_own_declared_level() -> None:
    steps = [
        classified_step(PiiGroup, ("raw",)),
        _derived(_Masking, "masked", "raw"),
        _derived(_Public, "after", "masked"),
    ]

    assert feature_classifications(steps, undeclared="public") == {
        "raw": "pii",
        "masked": "internal",
        "after": "internal",
    }


@pytest.mark.parametrize("group, expected", [(_Public, "public"), (InternalGroup, "internal")])
def test_a_masking_group_caps_the_level_of_its_input(group: type[FeatureGroup], expected: str) -> None:
    steps = [classified_step(group, ("raw",)), _derived(_Masking, "masked", "raw")]

    assert feature_classifications(steps, undeclared="public")["masked"] == expected


def test_a_masking_group_over_an_undeclared_input_caps_the_undeclared_level() -> None:
    steps = [classified_step(_Undeclared, ("raw",)), _derived(_Masking, "masked", "raw")]

    assert feature_classifications(steps, undeclared="public")["masked"] == "public"


@pytest.mark.parametrize(
    "reader, expected", [(_PublicReader, "public"), (_UndeclaredReader, "internal"), (_PiiReader, "internal")]
)
def test_a_masking_group_caps_the_level_of_its_reader(reader: type[BaseInputData], expected: str) -> None:
    steps = [classified_step(_Masking, ("a",), reader=reader)]

    assert feature_classifications(steps, undeclared="public") == {"a": expected}


def test_a_reader_declaring_nothing_is_unbounded_even_with_a_public_input() -> None:
    steps = [
        classified_step(_Public, ("raw",)),
        classified_step(_Masking, ("m",), {"m": ("raw",)}, reader=_UndeclaredReader),
    ]

    assert feature_classifications(steps, undeclared="public")["m"] == "internal"


def test_a_masking_root_step_without_reader_or_inputs_gets_its_own_level() -> None:
    result = feature_classifications([classified_step(_Masking, ("a",))], undeclared="public")

    assert result == {"a": "internal"}


def test_a_masking_output_depending_on_itself_raises() -> None:
    steps = [classified_step(PiiGroup, ("a",)), classified_step(_Masking, ("a",), {"a": ("a",)})]

    with pytest.raises(ValueError):
        feature_classifications(steps, undeclared="public")


def test_masking_without_its_own_level_lowers_nothing() -> None:
    steps = [classified_step(PiiGroup, ("raw",)), _derived(_MaskingWithoutLevel, "masked", "raw")]

    assert feature_classifications(steps, undeclared="public")["masked"] == "pii"


def test_masking_cannot_lower_below_a_reader_only_declaration() -> None:
    class _MaskingReaderOnly(FeatureGroup):
        masking = True

    steps = [classified_step(_MaskingReaderOnly, ("a",), reader=_PiiReader)]

    assert feature_classifications(steps, undeclared="public") == {"a": "pii"}


def test_a_masking_declared_attribute_without_the_class_attribute_does_not_lower() -> None:
    class _MaskingOnlyDeclared(FeatureGroup):
        @classmethod
        def declared_attributes(cls, features: Any) -> Mapping[str, Any]:
            return {CLASSIFICATION_KEY: "internal", MASKING_ATTRIBUTE: True}

    steps = [classified_step(PiiGroup, ("raw",)), _derived(_MaskingOnlyDeclared, "masked", "raw")]

    assert feature_classifications(steps, undeclared="public")["masked"] == "pii"


def test_the_per_feature_masking_option_does_not_lower() -> None:
    options = Options(context={MASKING_ATTRIBUTE: True})
    steps = [
        classified_step(PiiGroup, ("raw",)),
        classified_step(InternalGroup, ("d",), {"d": ("raw",)}, options=options),
    ]

    assert feature_classifications(steps, undeclared="public")["d"] == "pii"


def test_a_masking_value_that_is_not_true_does_not_lower() -> None:
    steps = [classified_step(PiiGroup, ("raw",)), _derived(_MaskingAsString, "masked", "raw")]

    assert feature_classifications(steps, undeclared="public")["masked"] == "pii"


def test_several_producers_of_one_name_give_the_most_restrictive() -> None:
    steps = [classified_step(InternalGroup, ("a",)), classified_step(PiiGroup, ("a",))]

    assert feature_classifications(steps, undeclared="public") == {"a": "pii"}


@pytest.mark.parametrize("group", [_Public, _Masking])
def test_an_input_without_a_producer_raises_naming_the_feature(group: type[FeatureGroup]) -> None:
    with pytest.raises(ValueError, match="ghost"):
        feature_classifications([_derived(group, "d", "ghost")], undeclared="public")


def test_an_unknown_declared_level_raises_naming_the_owner() -> None:
    with pytest.raises(ValueError, match="_Unknown|secret"):
        feature_classifications([classified_step(_Unknown, ("a",))], undeclared="public")


def test_an_invalid_undeclared_level_raises() -> None:
    with pytest.raises(ValueError):
        feature_classifications([classified_step(_Undeclared, ("a",))], undeclared="secret")


def test_a_cycle_raises() -> None:
    steps = [_derived(_Public, "a", "b"), _derived(_Public, "b", "a")]

    with pytest.raises(ValueError):
        feature_classifications(steps, undeclared="public")


def test_the_result_does_not_depend_on_step_order() -> None:
    steps = [
        classified_step(PiiGroup, ("raw",)),
        _derived(_Masking, "masked", "raw"),
        _derived(_Public, "after", "masked"),
    ]

    forward = feature_classifications(steps, undeclared="public")

    assert feature_classifications(list(reversed(steps)), undeclared="public") == forward


def test_the_steps_may_be_any_iterable() -> None:
    steps = [classified_step(PiiGroup, ("raw",)), _derived(_Public, "d", "raw")]

    assert feature_classifications(iter(steps), undeclared="public")["d"] == "pii"


@pytest.mark.parametrize("kind", ["join", "transform"])
def test_join_and_transform_steps_are_skipped(kind: str) -> None:
    steps = [classified_step(PiiGroup, ("a",)), classified_step(PiiGroup, ("joined",), kind=kind)]

    assert feature_classifications(steps, undeclared="public") == {"a": "pii"}


_CLASSIFICATION_LOGGER = "mloda.community.extenders.shared.classification"


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == _CLASSIFICATION_LOGGER and r.levelno == logging.WARNING]


def _tracker(undeclared: str | None = "public") -> RunClassifications:
    return RunClassifications(undeclared, "Owner")


class TestRunClassifications:
    def test_an_unknown_undeclared_level_raises_naming_the_levels(self) -> None:
        with pytest.raises(ValueError, match="pii"):
            RunClassifications("secret", "Owner")

    def test_off_stores_nothing(self) -> None:
        tracker = _tracker(None)

        tracker.start("run", [classified_step(PiiGroup, ("a",))], raise_on_error=True)

        assert tracker.levels("run") == {}
        assert tracker.step_level("run", ["a"]) is None

    def test_a_run_id_of_none_stores_nothing(self) -> None:
        tracker = _tracker()

        tracker.start(None, [classified_step(PiiGroup, ("a",))], raise_on_error=True)

        assert tracker.levels(None) == {}
        assert tracker.step_level(None, ["a"]) is None

    def test_levels_hold_the_effective_levels_of_the_run(self) -> None:
        tracker = _tracker()

        tracker.start("run", [classified_step(PiiGroup, ("raw",)), _derived(_Public, "d", "raw")], raise_on_error=False)

        assert tracker.levels("run") == {"raw": "pii", "d": "pii"}
        assert tracker.levels("other") == {}

    def test_step_level_is_the_most_restrictive_among_the_names(self) -> None:
        tracker = _tracker()
        tracker.start(
            "run", [classified_step(InternalGroup, ("a",)), classified_step(PiiGroup, ("b",))], raise_on_error=False
        )

        assert tracker.step_level("run", ["a", "b"]) == "pii"
        assert tracker.step_level("run", ["a"]) == "internal"

    def test_step_level_ignores_names_without_a_level(self) -> None:
        tracker = _tracker()
        tracker.start("run", [classified_step(InternalGroup, ("a",))], raise_on_error=False)

        assert tracker.step_level("run", ["ghost", "a"]) == "internal"

    def test_step_level_is_none_when_no_name_has_a_level(self) -> None:
        tracker = _tracker()
        tracker.start("run", [classified_step(InternalGroup, ("a",))], raise_on_error=False)

        assert tracker.step_level("run", ["ghost"]) is None
        assert tracker.step_level("other", ["a"]) is None

    def test_pop_forgets_the_run(self) -> None:
        tracker = _tracker()
        tracker.start("run", [classified_step(PiiGroup, ("a",))], raise_on_error=False)

        tracker.pop("run")
        tracker.pop(None)

        assert tracker.levels("run") == {}

    def test_a_resolution_failure_logs_one_warning_with_the_type_only_and_stores_nothing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        tracker = _tracker()

        with caplog.at_level(logging.WARNING, logger=_CLASSIFICATION_LOGGER):
            tracker.start("run", [classified_step(_Unknown, ("a",))], raise_on_error=False)

        records = _warnings(caplog)
        assert len(records) == 1
        message = records[0].getMessage()
        assert "Owner" in message
        assert "ValueError" in message
        assert "secret" not in message
        assert tracker.levels("run") == {}

    def test_a_resolution_failure_raises_when_raise_on_error(self) -> None:
        tracker = _tracker()

        with pytest.raises(ValueError):
            tracker.start("run", [classified_step(_Unknown, ("a",))], raise_on_error=True)

        assert tracker.levels("run") == {}

    def test_a_pickle_round_trip_keeps_undeclared_and_stored_levels(self) -> None:
        tracker = _tracker("internal")
        tracker.start("run", [classified_step(PiiGroup, ("a",))], raise_on_error=False)

        clone = pickle.loads(pickle.dumps(tracker))  # nosec

        assert clone.undeclared == "internal"
        assert clone.levels("run") == {"a": "pii"}
        clone.start("run2", [classified_step(_Undeclared, ("b",))], raise_on_error=False)
        assert clone.levels("run2") == {"b": "internal"}
        clone.pop("run")
        assert clone.levels("run") == {}
        assert tracker.levels("run") == {"a": "pii"}
