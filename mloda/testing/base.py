"""Base test utilities for mloda plugins."""

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import pytest
from mloda.provider import ComputeFramework, FeatureGroup
from mloda.user import Feature, FeatureName, Options


class _AllComputeFrameworks:
    """Sentinel type: the group declares no compute_framework_rule (it returns None, meaning any framework)."""

    def __repr__(self) -> str:
        return "ALL_COMPUTE_FRAMEWORKS"


ALL_COMPUTE_FRAMEWORKS = _AllComputeFrameworks()


class FeatureGroupTestBase:
    """Base class for testing FeatureGroup implementations.

    Declare the class attributes below in a ``Test*`` subclass; the inherited ``test_contract_*`` tests check them.
    ``expected_input_features`` is keyed by name, so two matching entries sharing a name share one expectation.
    """

    feature_group_class: ClassVar[type[FeatureGroup] | None] = None
    expected_compute_frameworks: ClassVar[set[type[ComputeFramework]] | _AllComputeFrameworks | None] = None
    matching_feature_names: ClassVar[Sequence[str | Feature] | None] = None
    non_matching_feature_names: ClassVar[Sequence[str | Feature] | None] = None
    expected_input_features: ClassVar[Mapping[str, set[str] | None] | None] = None

    def _unset(self, attr: str) -> Any:
        value = getattr(type(self), attr)
        if value is None:
            pytest.fail(f"{type(self).__name__}.{attr} is not set")
        return value

    def _group(self) -> type[FeatureGroup]:
        group: type[FeatureGroup] = self._unset("feature_group_class")
        return group

    @staticmethod
    def _name_and_options(item: str | Feature) -> tuple[str, Options]:
        if isinstance(item, str):
            return item, Options()
        return str(item.name), item.options

    def _matches(self, item: str | Feature) -> bool:
        name, options = self._name_and_options(item)
        try:
            return bool(self._group().match_feature_group_criteria(name, options))
        except ValueError:
            return False

    def test_contract_feature_group_class_is_a_feature_group(self) -> None:
        group = self._unset("feature_group_class")
        if not (isinstance(group, type) and issubclass(group, FeatureGroup)):
            pytest.fail(f"{type(self).__name__}.feature_group_class must be a FeatureGroup subclass, got {group!r}")

    def test_contract_compute_framework_rule(self) -> None:
        expected = self._unset("expected_compute_frameworks")
        actual = self._group().compute_framework_rule()
        label = f"{type(self).__name__}.expected_compute_frameworks"
        if expected is ALL_COMPUTE_FRAMEWORKS:
            if actual is not None:
                pytest.fail(f"{label} is ALL_COMPUTE_FRAMEWORKS but the rule returns {actual!r}")
        elif actual is None:
            pytest.fail(f"{label} is {expected!r} but the group declares no rule; use ALL_COMPUTE_FRAMEWORKS")
        elif actual != expected:
            pytest.fail(f"{label} is {expected!r} but the rule returns {actual!r}")

    def test_contract_matching_feature_names_match(self) -> None:
        names = self._unset("matching_feature_names")
        if len(names) == 0:
            pytest.fail(f"{type(self).__name__}.matching_feature_names must not be empty")
        offenders = [self._name_and_options(n)[0] for n in names if not self._matches(n)]
        if offenders:
            pytest.fail(f"{type(self).__name__}: names that do not match {self._group().__name__}: {offenders}")

    def test_contract_non_matching_feature_names_do_not_match(self) -> None:
        names = self._unset("non_matching_feature_names")
        offenders = [self._name_and_options(n)[0] for n in names if self._matches(n)]
        if offenders:
            pytest.fail(f"{type(self).__name__}: names that unexpectedly match {self._group().__name__}: {offenders}")

    def test_contract_input_features(self) -> None:
        names = self._unset("matching_feature_names")
        expectations = self._unset("expected_input_features")
        group = self._group()
        problems: list[str] = []
        for item in names:
            name, options = self._name_and_options(item)
            if name not in expectations:
                problems.append(f"{name}: no entry in {type(self).__name__}.expected_input_features")
                continue
            expected = expectations[name]
            try:
                returned = group().input_features(options, FeatureName(name))
            except NotImplementedError:
                returned = None
            actual = None if returned is None else {str(getattr(f, "name", f)) for f in returned}
            if actual != expected:
                problems.append(f"{name}: expected {expected!r}, got {actual!r}")
        if problems:
            pytest.fail("input_features mismatch: " + "; ".join(problems))
