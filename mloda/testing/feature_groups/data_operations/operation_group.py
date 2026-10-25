"""Reusable shape tests for a data-operations OperationFeatureGroup and its Implementations.

A family subclasses ``OperationGroupTestBase`` and fills the hooks; the concrete tests check
behaviour: one concrete group, one Implementation per framework, declared ``supported`` and
``divergent``, subtype resolution and the entry-point manifest.

Usage::

    class TestAggregationOperationGroup(OperationGroupTestBase):
        @classmethod
        def group_class(cls): ...
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from abc import ABC, abstractmethod
from typing import Any, ClassVar, NamedTuple

import pytest
from mloda.provider import FeatureChainParserMixin, FeatureGroup, FeatureSet, Implementation, OperationFeatureGroup
from mloda.user import Options

from mloda.testing.feature_groups.data_operations.mixins.case_parametrization import CaseParametrizationTestMixin


class ImplSpec(NamedTuple):
    """One Implementation of the group and what it must declare.

    ``module`` is the module that defines the Implementation. ``supported`` None means unrestricted.
    ``dependency`` is the optional package that gates the manifest entry.
    """

    id: str
    module: str
    name: str
    framework_module: str
    framework_name: str
    supported: frozenset[str] | None = None
    divergent: frozenset[str] = frozenset()
    dependency: str | None = None


class OperationGroupTestBase(CaseParametrizationTestMixin, ABC):
    """Abstract base: fill the hooks, inherit the behavioural shape tests."""

    # -- Hooks ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def group_class(cls) -> Any:
        """The OperationFeatureGroup under test."""

    @classmethod
    @abstractmethod
    def backend_package(cls) -> str:
        """Dotted package holding the Implementation modules and ``manifest``."""

    @classmethod
    @abstractmethod
    def implementations(cls) -> list[ImplSpec]:
        """Every shipped Implementation, with its expected ``supported`` and ``divergent``."""

    @classmethod
    @abstractmethod
    def subtype_key(cls) -> str | None:
        """The ``SUBTYPES`` key (e.g. ``aggregation_type``); None for a resolver-based declaration."""

    @classmethod
    @abstractmethod
    def loop_method(cls) -> str:
        """Name of the group's loop method (e.g. ``aggregate``)."""

    @classmethod
    @abstractmethod
    def compute_method(cls) -> str:
        """Name of the per-backend compute hook (e.g. ``_compute_group``); the group must not define it."""

    @classmethod
    @abstractmethod
    def probe_feature_set(cls) -> FeatureSet:
        """A valid feature set without an implementation, for the missing-implementation check."""

    @classmethod
    @abstractmethod
    def name_probes(cls) -> list[tuple[str, str]]:
        """(feature name, subtype) pairs resolved from the name alone."""

    @classmethod
    @abstractmethod
    def option_probes(cls) -> list[tuple[str, Options, str]]:
        """(feature name, config options, subtype) triples resolved from the config option."""

    @classmethod
    def alias_pairs(cls) -> list[tuple[str, str]]:
        """Subtypes that are aliases and so must be divergent together."""
        return []

    @classmethod
    def universe(cls) -> frozenset[str] | None:
        """The exact ``subtype_universe()``; ``frozenset()`` means no subtype dimension. Default: not pinned."""
        return None

    @classmethod
    def unresolved_probes(cls) -> list[tuple[str, Options]]:
        """(feature name, options) pairs that must resolve to no subtype."""
        return []

    @classmethod
    def mixins(cls) -> tuple[type, ...] | None:
        """The mixins that must directly follow the group in the MRO, before OperationFeatureGroup. Default: skip."""
        return None

    # -- Plumbing ---------------------------------------------------------------

    _case_fixtures: ClassVar[dict[str, str]] = {"impl_id": "implementation_ids"}

    @classmethod
    def implementation_ids(cls) -> list[str]:
        return [spec.id for spec in cls.implementations()]

    @classmethod
    def _spec(cls, impl_id: str) -> ImplSpec:
        return next(spec for spec in cls.implementations() if spec.id == impl_id)

    @classmethod
    def _implementation(cls, spec: ImplSpec) -> Any:
        """The Implementation class; skips only when its declared optional dependency is missing."""
        path = f"{cls.backend_package()}.{spec.module}"
        module = pytest.importorskip(path) if spec.dependency else importlib.import_module(path)
        return getattr(module, spec.name)

    # -- Group shape ------------------------------------------------------------

    def test_group_is_a_concrete_operation_group(self) -> None:
        group = self.group_class()
        assert issubclass(group, OperationFeatureGroup)
        assert not inspect.isabstract(group)

    def test_group_matches_before_the_bare_feature_group(self) -> None:
        """The subtype matcher mixin must precede FeatureGroup in the MRO, whoever defines the matcher."""
        mro = self.group_class().__mro__
        assert mro.index(FeatureChainParserMixin) < mro.index(FeatureGroup)

    def test_group_declares_the_subtype_key(self) -> None:
        if self.universe() == frozenset():
            assert self.group_class().SUBTYPES is None
            return
        declaration = self.group_class().SUBTYPES
        assert declaration is not None
        assert declaration.key == self.subtype_key()
        if self.subtype_key() is None:
            assert declaration.resolver is not None

    def test_group_mro_puts_the_mixins_before_the_operation_group(self) -> None:
        mixins = self.mixins()
        if mixins is None:
            pytest.skip("no mixins declared")
        mro = self.group_class().__mro__
        assert mro[1 : len(mixins) + 3] == (*mixins, OperationFeatureGroup, FeatureGroup)

    def test_group_owns_the_loop_and_no_per_framework_compute(self) -> None:
        assert callable(getattr(self.group_class(), self.loop_method()))
        assert not hasattr(self.group_class(), self.compute_method())

    def test_group_needs_an_implementation_on_the_feature_set(self) -> None:
        """calculate_feature dispatches to the step's implementation; none on the feature set is an error."""
        from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator

        with pytest.raises(ValueError, match="no implementation"):
            self.group_class().calculate_feature(PyArrowDataOpsTestDataCreator.create(), self.probe_feature_set())

    def test_implementation_names_its_pair(self, impl_id: str) -> None:
        spec = self._spec(impl_id)
        impl = self._implementation(spec)
        framework = getattr(pytest.importorskip(spec.framework_module), spec.framework_name)
        assert issubclass(impl, Implementation)
        assert not issubclass(impl, FeatureGroup)
        assert impl.feature_group is self.group_class()
        assert impl.compute_framework is framework
        assert impl.__module__ == f"{self.backend_package()}.{spec.module}"

    def test_implementation_keeps_its_compute_hook(self, impl_id: str) -> None:
        spec = self._spec(impl_id)
        assert callable(getattr(self._implementation(spec), self.compute_method()))

    # -- Supported and divergent ------------------------------------------------

    def test_supported_is_the_declared_set(self, impl_id: str) -> None:
        spec = self._spec(impl_id)
        assert self._implementation(spec).supported == spec.supported

    def test_supported_stays_inside_the_universe(self, impl_id: str) -> None:
        """A typo in supported cannot silently hide a real subtype."""
        spec = self._spec(impl_id)
        supported = self._implementation(spec).supported
        assert supported is None or supported <= self.group_class().subtype_universe()

    def test_divergent_is_the_declared_set(self, impl_id: str) -> None:
        spec = self._spec(impl_id)
        assert set(self._implementation(spec).divergent) == spec.divergent

    def test_divergent_is_inside_the_universe_with_a_reason(self, impl_id: str) -> None:
        spec = self._spec(impl_id)
        divergent = self._implementation(spec).divergent
        assert set(divergent) <= self.group_class().subtype_universe()
        assert all(isinstance(reason, str) and reason for reason in divergent.values())

    def test_divergent_aliases_diverge_together(self, impl_id: str) -> None:
        """An alias shares its target's behaviour, so a backend that diverges on one diverges on the other."""
        if not self.alias_pairs():
            pytest.skip("no alias pairs declared")
        spec = self._spec(impl_id)
        divergent = self._implementation(spec).divergent
        for first, second in self.alias_pairs():
            assert (first in divergent) == (second in divergent)

    def test_universe_is_the_declared_set(self) -> None:
        if self.universe() is None:
            pytest.skip("no universe declared")
        assert self.group_class().subtype_universe() == self.universe()

    # -- Subtype resolution -----------------------------------------------------

    def test_resolves_from_the_feature_name(self) -> None:
        for name, subtype in self.name_probes():
            assert self.group_class().resolve_subtype(name, Options()) == subtype

    def test_resolves_from_the_config_option(self) -> None:
        for name, options, subtype in self.option_probes():
            assert self.group_class().resolve_subtype(name, options) == subtype

    def test_unresolvable_probes_resolve_to_no_subtype(self) -> None:
        for name, options in self.unresolved_probes():
            assert self.group_class().resolve_subtype(name, options) is None

    # -- Manifest ---------------------------------------------------------------

    def test_manifest_lists_the_group_and_every_installed_implementation(self) -> None:
        feature_groups = importlib.import_module(f"{self.backend_package()}.manifest").FEATURE_GROUPS
        group = self.group_class()
        assert group in feature_groups
        listed = {cls.__name__ for cls in feature_groups if issubclass(cls, Implementation)}
        expected = {
            spec.name
            for spec in self.implementations()
            if spec.dependency is None or importlib.util.find_spec(spec.dependency) is not None
        }
        assert listed == expected
        implementations: list[Any] = [cls for cls in feature_groups if cls is not group]
        assert all(cls.feature_group is group for cls in implementations)
