"""Tests for mloda.testing package."""

import subprocess  # nosec
import sys
from typing import Any

import pytest
from mloda.provider import ComputeFramework, FeatureGroup
from mloda.steward import Extender, ExtenderHook
from mloda.user import Feature, FeatureName, Options

from mloda.testing import base as base_module
from mloda.testing.base import FeatureGroupTestBase
from mloda.testing.extenders.contract import ExtenderContractTestMixin


def test_feature_group_test_base_import() -> None:
    """Verify FeatureGroupTestBase can be imported."""
    assert FeatureGroupTestBase is not None


def test_import_does_not_load_pytest_outside_pytest() -> None:
    """Importing mloda.testing must not import pytest in a process where pytest is not loaded."""
    result = subprocess.run(  # nosec
        [sys.executable, "-c", "import sys, mloda.testing; print('pytest' in sys.modules)"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_contract_mixin_assert_reports_compared_values() -> None:
    """Self-test: a failing contract assert should show the compared values, not an empty message."""

    class _OffByOneExtender(Extender):
        def __init__(self) -> None:
            self.raise_on_error = True

        def wraps(self) -> set[ExtenderHook]:
            return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

        def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs) + 1

    class _Host(ExtenderContractTestMixin):
        def make_extender(self, *, raise_on_error: bool | None = None) -> _OffByOneExtender:
            return _OffByOneExtender()

    with pytest.raises(AssertionError, match=r"assert 8 == 7"):
        _Host().test_contract_call_returns_wrapped_result_unchanged()


# -------------------------------------------------------------------------------------------
# FeatureGroupTestBase contract self-tests
# -------------------------------------------------------------------------------------------


class _RootGroup(FeatureGroup):
    """Matches only its own class name; has no input features (core default raises NotImplementedError)."""


class _DerivedGroup(FeatureGroup):
    """Matches names ending in ``__derived``, rejects ``bad__derived`` with ValueError; reads the prefix as input."""

    @classmethod
    def match_feature_group_criteria(
        cls, feature_name: FeatureName | str, options: Options, data_access_collection: Any = None
    ) -> bool:
        name = str(feature_name)
        if name == "bad__derived":
            raise ValueError("rejected by a strict spec")
        return name.endswith("__derived")

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(str(feature_name).removesuffix("__derived"))}


class _StrInputGroup(_DerivedGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Any:
        return {str(feature_name).removesuffix("__derived")}


class _FrameworkGroup(_DerivedGroup):
    @classmethod
    def compute_framework_rule(cls) -> Any:
        return {ComputeFramework}


def _all_cfw() -> Any:
    return getattr(base_module, "ALL_COMPUTE_FRAMEWORKS", None)


def _full_host(**overrides: Any) -> Any:
    """A fully declared, passing host for ``_DerivedGroup``; ``overrides`` replace declared attributes."""
    attrs: dict[str, Any] = {
        "feature_group_class": _DerivedGroup,
        "expected_compute_frameworks": _all_cfw(),
        "matching_feature_names": ["a__derived"],
        "non_matching_feature_names": ["plain"],
        "expected_input_features": {"a__derived": {"a"}},
    }
    attrs.update(overrides)
    return type("_Host", (FeatureGroupTestBase,), attrs)()


_UNSET_CASES = [
    ("feature_group_class", "test_contract_feature_group_class_is_a_feature_group"),
    ("expected_compute_frameworks", "test_contract_compute_framework_rule"),
    ("matching_feature_names", "test_contract_matching_feature_names_match"),
    ("non_matching_feature_names", "test_contract_non_matching_feature_names_do_not_match"),
    ("expected_input_features", "test_contract_input_features"),
]


def test_contract_base_declares_every_attribute_unset() -> None:
    for attr, _ in _UNSET_CASES:
        assert getattr(FeatureGroupTestBase, attr) is None


def test_contract_sentinel_has_a_clear_repr() -> None:
    assert "ALL_COMPUTE_FRAMEWORKS" in repr(_all_cfw())


@pytest.mark.parametrize(("attr", "method"), _UNSET_CASES)
def test_contract_unset_attribute_fails_naming_the_attribute(attr: str, method: str) -> None:
    host = _full_host(**{attr: None})
    with pytest.raises(pytest.fail.Exception, match=rf"_Host\.{attr}"):
        getattr(host, method)()


def test_contract_fully_declared_toy_passes_all_five() -> None:
    host = _full_host()
    host.test_contract_feature_group_class_is_a_feature_group()
    host.test_contract_compute_framework_rule()
    host.test_contract_matching_feature_names_match()
    host.test_contract_non_matching_feature_names_do_not_match()
    host.test_contract_input_features()


def test_contract_feature_group_class_must_be_a_feature_group() -> None:
    host = _full_host(feature_group_class=dict)
    with pytest.raises(pytest.fail.Exception, match=r"_Host\.feature_group_class"):
        host.test_contract_feature_group_class_is_a_feature_group()


def test_contract_missing_rule_without_sentinel_fails() -> None:
    host = _full_host(expected_compute_frameworks={ComputeFramework})
    with pytest.raises(pytest.fail.Exception, match=r"_Host\.expected_compute_frameworks"):
        host.test_contract_compute_framework_rule()


def test_contract_missing_rule_with_sentinel_passes() -> None:
    _full_host().test_contract_compute_framework_rule()


def test_contract_sentinel_fails_when_the_group_declares_a_rule() -> None:
    host = _full_host(feature_group_class=_FrameworkGroup)
    with pytest.raises(pytest.fail.Exception, match=r"_Host\.expected_compute_frameworks"):
        host.test_contract_compute_framework_rule()


def test_contract_matching_rule_passes() -> None:
    host = _full_host(feature_group_class=_FrameworkGroup, expected_compute_frameworks={ComputeFramework})
    host.test_contract_compute_framework_rule()


def test_contract_wrong_rule_fails() -> None:
    host = _full_host(feature_group_class=_FrameworkGroup, expected_compute_frameworks={FeatureGroup})
    with pytest.raises(pytest.fail.Exception, match=r"_Host\.expected_compute_frameworks"):
        host.test_contract_compute_framework_rule()


def test_contract_matching_names_must_not_be_empty() -> None:
    host = _full_host(matching_feature_names=[])
    with pytest.raises(pytest.fail.Exception, match=r"_Host\.matching_feature_names"):
        host.test_contract_matching_feature_names_match()


def test_contract_offending_matching_name_is_listed() -> None:
    host = _full_host(matching_feature_names=["a__derived", "offender_one", Feature("offender_two", Options())])
    with pytest.raises(pytest.fail.Exception) as excinfo:
        host.test_contract_matching_feature_names_match()
    message = str(excinfo.value)
    assert "offender_one" in message
    assert "offender_two" in message
    assert "a__derived" not in message


def test_contract_matching_name_rejected_by_value_error_fails() -> None:
    host = _full_host(matching_feature_names=["bad__derived"])
    with pytest.raises(pytest.fail.Exception, match="bad__derived"):
        host.test_contract_matching_feature_names_match()


def test_contract_offending_non_matching_name_is_listed() -> None:
    host = _full_host(non_matching_feature_names=["plain", "x__derived", Feature("y__derived", Options())])
    with pytest.raises(pytest.fail.Exception) as excinfo:
        host.test_contract_non_matching_feature_names_do_not_match()
    message = str(excinfo.value)
    assert "x__derived" in message
    assert "y__derived" in message
    assert "plain" not in message


def test_contract_value_error_counts_as_non_match() -> None:
    host = _full_host(non_matching_feature_names=["bad__derived"])
    host.test_contract_non_matching_feature_names_do_not_match()


def test_contract_empty_non_matching_names_is_an_explicit_none() -> None:
    _full_host(non_matching_feature_names=[]).test_contract_non_matching_feature_names_do_not_match()


def test_contract_wrong_input_expectation_fails_with_name_expected_and_actual() -> None:
    host = _full_host(expected_input_features={"a__derived": {"zzz"}})
    with pytest.raises(pytest.fail.Exception) as excinfo:
        host.test_contract_input_features()
    message = str(excinfo.value)
    assert "a__derived" in message
    assert "zzz" in message
    assert "'a'" in message


def test_contract_missing_input_expectation_fails() -> None:
    host = _full_host(expected_input_features={})
    with pytest.raises(pytest.fail.Exception, match="a__derived"):
        host.test_contract_input_features()


def test_contract_none_expectation_fails_when_group_has_inputs() -> None:
    host = _full_host(expected_input_features={"a__derived": None})
    with pytest.raises(pytest.fail.Exception, match="a__derived"):
        host.test_contract_input_features()


def test_contract_input_features_accept_str_items() -> None:
    _full_host(feature_group_class=_StrInputGroup).test_contract_input_features()


def test_contract_not_implemented_error_counts_as_root_group() -> None:
    host = _full_host(
        feature_group_class=_RootGroup,
        matching_feature_names=["_RootGroup"],
        non_matching_feature_names=["plain"],
        expected_input_features={"_RootGroup": None},
    )
    host.test_contract_matching_feature_names_match()
    host.test_contract_non_matching_feature_names_do_not_match()
    host.test_contract_input_features()


def test_contract_root_group_fails_when_inputs_are_expected() -> None:
    host = _full_host(
        feature_group_class=_RootGroup,
        matching_feature_names=["_RootGroup"],
        expected_input_features={"_RootGroup": {"a"}},
    )
    with pytest.raises(pytest.fail.Exception, match="_RootGroup"):
        host.test_contract_input_features()


class _EmptySetGroup(_RootGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature]:
        return set()


def test_contract_empty_input_set_counts_as_root_group() -> None:
    host = _full_host(
        feature_group_class=_EmptySetGroup,
        matching_feature_names=["_EmptySetGroup"],
        expected_input_features={"_EmptySetGroup": None},
    )
    host.test_contract_input_features()


def test_contract_matcher_receives_feature_name_objects() -> None:
    seen: list[type] = []

    class _Recording(_DerivedGroup):
        @classmethod
        def match_feature_group_criteria(
            cls, feature_name: FeatureName | str, options: Options, data_access_collection: Any = None
        ) -> bool:
            seen.append(type(feature_name))
            return str(feature_name).endswith("__derived")

    _full_host(feature_group_class=_Recording).test_contract_matching_feature_names_match()

    assert seen == [FeatureName]
