"""Direct unit tests for mloda.testing.feature_set (build_feature_set)."""

from __future__ import annotations

from typing import Any, ClassVar

from mloda.provider import FeatureGroup, FeatureSet, property_spec
from mloda.user import Feature, FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.testing.feature_set import build_feature_set


class _ScaledToy(FeatureGroup):
    """Toy group with one defaulted option."""

    PROPERTY_MAPPING: ClassVar[dict[str, Any]] = {"factor": property_spec("Scale factor", default=2)}


def test_str_argument_becomes_a_feature_with_that_name() -> None:
    """A str is wrapped as Feature(name)."""
    fs = build_feature_set(None, "alpha")

    assert isinstance(fs, FeatureSet)
    (feature,) = fs.features
    assert isinstance(feature, Feature)
    assert str(feature.name) == "alpha"


def test_name_of_one_feature_is_a_feature_name_not_a_mock() -> None:
    """name_of_one_feature is a real FeatureName, equal to the str it was built from."""
    fs = build_feature_set(None, "alpha")

    assert isinstance(fs.name_of_one_feature, FeatureName)
    assert str(fs.name_of_one_feature) == "alpha"


def test_feature_arguments_are_kept_with_their_options() -> None:
    """A Feature argument keeps its name and options."""
    feature = Feature("beta", Options(context={"k": "v"}))

    fs = build_feature_set(None, feature)

    assert {str(f.name) for f in fs.features} == {"beta"}
    assert fs.get_options_key("k") == "v"


def test_several_features_in_one_call() -> None:
    """Mixed str and Feature arguments all land in the set."""
    fs = build_feature_set(None, "a", Feature("b"), "c")

    assert {str(f.name) for f in fs.features} == {"a", "b", "c"}


def test_option_default_is_materialized_for_the_given_group() -> None:
    """The group's PROPERTY_MAPPING default shows up in the feature options."""
    fs = build_feature_set(_ScaledToy, "x")

    (feature,) = fs.features
    assert feature.options.get("factor") == 2
    assert fs.get_options_key("factor") == 2


def test_no_default_is_materialized_without_a_group() -> None:
    """With feature_group None the defaults are skipped."""
    fs = build_feature_set(None, "x")

    (feature,) = fs.features
    assert feature.options.get("factor") is None


def test_explicit_option_value_is_not_overwritten_by_the_default() -> None:
    """An explicitly set value wins over the default."""
    fs = build_feature_set(_ScaledToy, Feature("x", Options(context={"factor": 5})))

    assert fs.get_options_key("factor") == 5


def test_compute_framework_wires_mask_and_filter_engines() -> None:
    """The framework's mask and filter engines are set on the FeatureSet as a run does."""
    assert PyArrowTable.mask_engine() is not None

    fs = build_feature_set(None, "x", compute_framework=PyArrowTable)

    expected_filter_engine: Any = PyArrowTable().filter_engine()
    assert fs.mask_engine is PyArrowTable.mask_engine()
    assert fs.filter_engine() is expected_filter_engine


def test_mask_engine_is_none_without_a_compute_framework() -> None:
    """Without a framework the mask engine stays unset."""
    fs = build_feature_set(None, "x")

    assert fs.mask_engine is None
