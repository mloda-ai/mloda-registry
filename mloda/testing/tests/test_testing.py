"""Tests for mloda.testing package."""

import subprocess  # nosec
import sys
from typing import Any

import pytest
from mloda.provider import ComputeFramework, FeatureGroup
from mloda.steward import Extender, ExtenderHook
from mloda.user import Feature, FeatureName, Options

from mloda.testing import base as base_module
from mloda.testing.base import FeatureGroupContractTestMixin, FeatureGroupTestBase
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


class _EmptySetGroup(_RootGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature]:
        return set()


_M_CLASS = "test_contract_feature_group_class_is_a_feature_group"
_M_RULE = "test_contract_compute_framework_rule"
_M_MATCH = "test_contract_matching_feature_names_match"
_M_NON_MATCH = "test_contract_non_matching_feature_names_do_not_match"
_M_INPUTS = "test_contract_input_features"

_FAIL_CASES: list[Any] = [
    *[pytest.param({a: None}, m, (f"_Host.{a}", f"contract_{a}()"), (), id=f"unset_{a}") for a, m in _UNSET_CASES],
    pytest.param(
        {"feature_group_class": dict},
        _M_CLASS,
        ("_Host.feature_group_class",),
        (),
        id="feature_group_class_must_be_a_feature_group",
    ),
    pytest.param(
        {"expected_compute_frameworks": {ComputeFramework}},
        _M_RULE,
        ("_Host.expected_compute_frameworks",),
        (),
        id="missing_rule_without_sentinel_fails",
    ),
    pytest.param(
        {"feature_group_class": _FrameworkGroup},
        _M_RULE,
        ("_Host.expected_compute_frameworks",),
        (),
        id="sentinel_fails_when_the_group_declares_a_rule",
    ),
    pytest.param(
        {"feature_group_class": _FrameworkGroup, "expected_compute_frameworks": {FeatureGroup}},
        _M_RULE,
        ("_Host.expected_compute_frameworks",),
        (),
        id="wrong_rule_fails",
    ),
    pytest.param(
        {"matching_feature_names": []},
        _M_MATCH,
        ("_Host.matching_feature_names",),
        (),
        id="matching_names_must_not_be_empty",
    ),
    pytest.param(
        {"matching_feature_names": ["a__derived", "offender_one", Feature("offender_two", Options())]},
        _M_MATCH,
        ("offender_one", "offender_two"),
        ("a__derived",),
        id="offending_matching_name_is_listed",
    ),
    pytest.param(
        {"matching_feature_names": ["bad__derived"]},
        _M_MATCH,
        ("bad__derived",),
        (),
        id="matching_name_rejected_by_value_error_fails",
    ),
    pytest.param(
        {"non_matching_feature_names": ["plain", "x__derived", Feature("y__derived", Options())]},
        _M_NON_MATCH,
        ("x__derived", "y__derived"),
        ("plain",),
        id="offending_non_matching_name_is_listed",
    ),
    pytest.param(
        {"expected_input_features": {"a__derived": {"zzz"}}},
        _M_INPUTS,
        ("a__derived", "zzz", "'a'"),
        (),
        id="wrong_input_expectation_fails_with_name_expected_and_actual",
    ),
    pytest.param({"expected_input_features": {}}, _M_INPUTS, ("a__derived",), (), id="missing_input_expectation_fails"),
    pytest.param(
        {"expected_input_features": {"a__derived": None}},
        _M_INPUTS,
        ("a__derived",),
        (),
        id="none_expectation_fails_when_group_has_inputs",
    ),
    pytest.param(
        {
            "feature_group_class": _RootGroup,
            "matching_feature_names": ["_RootGroup"],
            "expected_input_features": {"_RootGroup": {"a"}},
        },
        _M_INPUTS,
        ("_RootGroup",),
        (),
        id="root_group_fails_when_inputs_are_expected",
    ),
]


@pytest.mark.parametrize(("overrides", "method", "present", "absent"), _FAIL_CASES)
def test_contract_failure_messages(
    overrides: dict[str, Any], method: str, present: tuple[str, ...], absent: tuple[str, ...]
) -> None:
    with pytest.raises(pytest.fail.Exception) as excinfo:
        getattr(_full_host(**overrides), method)()
    message = str(excinfo.value)
    for text in present:
        assert text in message
    for text in absent:
        assert text not in message


_ALL_FIVE = (_M_CLASS, _M_RULE, _M_MATCH, _M_NON_MATCH, _M_INPUTS)

_PASS_CASES: list[Any] = [
    pytest.param({}, _ALL_FIVE, id="fully_declared_toy_passes_all_five"),
    pytest.param(
        {"feature_group_class": _FrameworkGroup, "expected_compute_frameworks": {ComputeFramework}},
        (_M_RULE,),
        id="matching_rule_passes",
    ),
    pytest.param(
        {"non_matching_feature_names": ["bad__derived"]}, (_M_NON_MATCH,), id="value_error_counts_as_non_match"
    ),
    pytest.param(
        {"non_matching_feature_names": []}, (_M_NON_MATCH,), id="empty_non_matching_names_is_an_explicit_none"
    ),
    pytest.param({"feature_group_class": _StrInputGroup}, (_M_INPUTS,), id="input_features_accept_str_items"),
    pytest.param(
        {
            "feature_group_class": _RootGroup,
            "matching_feature_names": ["_RootGroup"],
            "non_matching_feature_names": ["plain"],
            "expected_input_features": {"_RootGroup": None},
        },
        (_M_MATCH, _M_NON_MATCH, _M_INPUTS),
        id="not_implemented_error_counts_as_root_group",
    ),
    pytest.param(
        {
            "feature_group_class": _EmptySetGroup,
            "matching_feature_names": ["_EmptySetGroup"],
            "expected_input_features": {"_EmptySetGroup": None},
        },
        (_M_INPUTS,),
        id="empty_input_set_counts_as_root_group",
    ),
]


@pytest.mark.parametrize(("overrides", "methods"), _PASS_CASES)
def test_contract_passes(overrides: dict[str, Any], methods: tuple[str, ...]) -> None:
    host = _full_host(**overrides)
    for method in methods:
        getattr(host, method)()


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


# -------------------------------------------------------------------------------------------
# FeatureGroupContractTestMixin self-tests
# -------------------------------------------------------------------------------------------


def test_contract_mixin_host_with_hooks_passes_all_five() -> None:
    class _Host(FeatureGroupContractTestMixin):
        def contract_feature_group_class(self) -> Any:
            return _DerivedGroup

        def contract_expected_compute_frameworks(self) -> Any:
            return _all_cfw()

        def contract_matching_feature_names(self) -> Any:
            return ["a__derived"]

        def contract_non_matching_feature_names(self) -> Any:
            return ["plain"]

        def contract_expected_input_features(self) -> Any:
            return {"a__derived": {"a"}}

    host = _Host()
    for method in _ALL_FIVE:
        getattr(host, method)()
