"""Contract for a community extender package whose runtime dependency is optional."""

from __future__ import annotations

import importlib
import logging
import sys
import types

import pytest

from mloda.testing.import_isolation import block_root, evict_package, evict_root


class OptionalDependencyPackageTestMixin:
    """Manifest resilience contract. Host declares package, root, distribution, extra, extender_name,
    extender_module, broken_module, broken_name, transitive_dependency: mloda's entry-point loader must
    tolerate not only a missing ``root`` (the manifest degrades to an empty EXTENDERS list), but also a
    ``root`` that is installed yet unusable. Any import error unrelated to ``root`` must still propagate
    unchanged.

    ``broken_module`` names a real submodule of ``root`` from which ``broken_name`` is imported by the
    extender module; ``transitive_dependency`` names a real third-party dependency of ``root`` itself
    (``None`` when the host has no safe transitive dependency to poison, in which case the corresponding
    tests are skipped).
    """

    package: str
    root: str
    distribution: str
    extra: str
    extender_name: str
    extender_module: str
    broken_module: str
    broken_name: str
    transitive_dependency: str | None

    def test_manifest_lists_the_extender_when_installed(self) -> None:
        manifest = importlib.import_module(f"{self.package}.manifest")
        extender_module = importlib.import_module(f"{self.package}.{self.extender_module}")
        extender = getattr(extender_module, self.extender_name)

        assert manifest.EXTENDERS == [extender]

    def test_manifest_is_empty_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        block_root(monkeypatch, self.root)
        evict_package(monkeypatch, self.package)

        module = importlib.import_module(f"{self.package}.manifest")

        assert module.EXTENDERS == []

    def test_manifest_reraises_unrelated_import_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, f"{self.package}.{self.extender_module}", None)
        monkeypatch.delitem(sys.modules, f"{self.package}.manifest", raising=False)

        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"{self.package}.manifest")

    def test_manifest_reraises_unrelated_plain_import_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A plain ImportError (not ModuleNotFoundError) from our own submodule, unrelated to ``root``:
        the widened except clause must not swallow it too."""
        fake_module = types.ModuleType(f"{self.package}.{self.extender_module}")
        monkeypatch.setitem(sys.modules, f"{self.package}.{self.extender_module}", fake_module)
        monkeypatch.delitem(sys.modules, f"{self.package}.manifest", raising=False)

        with pytest.raises(ImportError) as info:
            importlib.import_module(f"{self.package}.manifest")

        assert not isinstance(info.value, ModuleNotFoundError)

    def test_package_imports_without_dependency_and_hides_the_extender(self, monkeypatch: pytest.MonkeyPatch) -> None:
        block_root(monkeypatch, self.root)
        evict_package(monkeypatch, self.package)

        module = importlib.import_module(self.package)

        assert self.extender_name not in vars(module)

        with pytest.raises(ImportError):
            getattr(module, self.extender_name)

    def test_missing_dependency_import_error_names_distribution_and_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        block_root(monkeypatch, self.root)
        evict_package(monkeypatch, self.package)

        with pytest.raises(ImportError) as info:
            getattr(importlib.import_module(self.package), self.extender_name)

        assert self.distribution in str(info.value)
        assert self.extra in str(info.value)

        module = importlib.import_module(self.package)
        assert getattr(module, "some_other_name", None) is None

    def test_manifest_logs_when_dependency_is_missing(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        block_root(monkeypatch, self.root)
        evict_package(monkeypatch, self.package)

        with caplog.at_level(logging.INFO):
            importlib.import_module(f"{self.package}.manifest")

        matching = [
            record
            for record in caplog.records
            if record.name.startswith("mloda.community.extenders.")
            and record.levelno >= logging.INFO
            and self.extra in record.getMessage()
        ]
        assert matching, f"no INFO log named the {self.extra} extra"

        warnings = [
            record
            for record in caplog.records
            if record.name.startswith("mloda.community.extenders.") and record.levelno == logging.WARNING
        ]
        assert not warnings, (
            f"unexpected WARNING when {self.root} is genuinely absent (should be INFO only): "
            f"{[r.getMessage() for r in warnings]}"
        )

    def test_package_hides_extender_and_chains_cause_when_dependency_is_broken(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, self.broken_module, None)
        evict_package(monkeypatch, self.package)

        module = importlib.import_module(self.package)

        assert self.extender_name not in vars(module)

        with pytest.raises(ImportError) as info:
            getattr(module, self.extender_name)

        assert self.distribution in str(info.value)
        assert self.extra in str(info.value)
        assert info.value.__cause__ is not None

    def test_manifest_is_empty_when_dependency_module_is_broken(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``root`` is installed but one of ITS OWN submodules is absent, as if too old a version."""
        monkeypatch.setitem(sys.modules, self.broken_module, None)
        evict_package(monkeypatch, self.package)

        module = importlib.import_module(f"{self.package}.manifest")

        assert module.EXTENDERS == []

    def test_manifest_logs_warning_when_dependency_module_is_broken(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, self.broken_module, None)
        evict_package(monkeypatch, self.package)

        with caplog.at_level(logging.INFO):
            importlib.import_module(f"{self.package}.manifest")

        matching = [
            record
            for record in caplog.records
            if record.name.startswith("mloda.community.extenders.")
            and record.levelno == logging.WARNING
            and self.distribution in record.getMessage()
            and self.extra in record.getMessage()
            and self.broken_module in record.getMessage()
        ]
        assert matching, (
            f"no WARNING log naming the {self.distribution} distribution, the {self.extra} extra, and "
            f"the real {self.broken_module} failure"
        )

    def test_manifest_is_empty_when_dependency_name_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``root`` is installed but too old to expose ``broken_name`` inside ``broken_module``: unlike a
        missing submodule, this raises a plain ImportError rather than a ModuleNotFoundError."""
        monkeypatch.delattr(importlib.import_module(self.broken_module), self.broken_name)
        evict_package(monkeypatch, self.package)

        module = importlib.import_module(f"{self.package}.manifest")

        assert module.EXTENDERS == []

    def test_manifest_logs_warning_when_dependency_name_is_missing(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delattr(importlib.import_module(self.broken_module), self.broken_name)
        evict_package(monkeypatch, self.package)

        with caplog.at_level(logging.INFO):
            importlib.import_module(f"{self.package}.manifest")

        matching = [
            record
            for record in caplog.records
            if record.name.startswith("mloda.community.extenders.")
            and record.levelno == logging.WARNING
            and self.distribution in record.getMessage()
            and self.extra in record.getMessage()
            and self.broken_name in record.getMessage()
        ]
        assert matching, (
            f"no WARNING log naming the {self.distribution} distribution, the {self.extra} extra, and "
            f"the real missing {self.broken_name} name"
        )

    def test_manifest_is_empty_when_transitive_dependency_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``root`` is installed, but one of ITS OWN transitive dependencies is missing: the failure
        surfaces from a frame inside ``root``'s own code, not from ``root`` itself by name."""
        if self.transitive_dependency is None:
            pytest.skip("no transitive_dependency declared")

        evict_root(monkeypatch, self.root)
        monkeypatch.setitem(sys.modules, self.transitive_dependency, None)
        evict_package(monkeypatch, self.package)

        module = importlib.import_module(f"{self.package}.manifest")

        assert module.EXTENDERS == []

    def test_manifest_logs_warning_when_transitive_dependency_is_missing(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if self.transitive_dependency is None:
            pytest.skip("no transitive_dependency declared")

        evict_root(monkeypatch, self.root)
        monkeypatch.setitem(sys.modules, self.transitive_dependency, None)
        evict_package(monkeypatch, self.package)

        with caplog.at_level(logging.INFO):
            importlib.import_module(f"{self.package}.manifest")

        matching = [
            record
            for record in caplog.records
            if record.name.startswith("mloda.community.extenders.")
            and record.levelno == logging.WARNING
            and self.distribution in record.getMessage()
            and self.extra in record.getMessage()
            and self.transitive_dependency in record.getMessage()
        ]
        assert matching, (
            f"no WARNING log naming the {self.distribution} distribution, the {self.extra} extra, and "
            f"the real missing {self.transitive_dependency} transitive dependency"
        )

    def test_optional_dependency_helper_swallows_dependency_named_error(self) -> None:
        optional_dependency = importlib.import_module(f"{self.package}._optional_dependency")
        exc = ImportError("boom", name=f"{self.root}.x")

        assert optional_dependency.reraise_unless_optional(exc, self.root) is None

    def test_optional_dependency_helper_reraises_named_error_outside_root(self) -> None:
        optional_dependency = importlib.import_module(f"{self.package}._optional_dependency")
        exc = ImportError("boom", name="mloda.something")

        with pytest.raises(ImportError) as info:
            optional_dependency.reraise_unless_optional(exc, self.root)

        assert info.value is exc

    def test_optional_dependency_helper_reraises_nameless_error(self) -> None:
        optional_dependency = importlib.import_module(f"{self.package}._optional_dependency")
        exc = ImportError("boom")

        with pytest.raises(ImportError) as info:
            optional_dependency.reraise_unless_optional(exc, self.root)

        assert info.value is exc

    def test_blocking_tests_leave_no_degraded_module_behind(self) -> None:
        """A prior test blocking the dependency must not leak its degraded module into later imports."""
        root_package = "mloda.community.extenders"
        leaf_attr = self.package.rsplit(".", 1)[1]
        manifest_name = f"{self.package}.manifest"

        if self.package not in sys.modules:
            importlib.import_module(self.package)

        with pytest.MonkeyPatch.context() as pre_mp:
            if manifest_name in sys.modules:
                pre_mp.delitem(sys.modules, manifest_name)

            before_package = sys.modules.get(self.package)
            before_manifest = sys.modules.get(manifest_name)

            with pytest.MonkeyPatch.context() as mp:
                block_root(mp, self.root)
                evict_package(mp, self.package)
                importlib.import_module(manifest_name)

            assert sys.modules.get(manifest_name) is before_manifest
            assert sys.modules.get(self.package) is before_package

            extender_module = importlib.import_module(f"{self.package}.{self.extender_module}")
            extender = getattr(extender_module, self.extender_name)

            module = importlib.import_module(manifest_name)
            assert module.EXTENDERS == [extender]

        restored_leaf = getattr(importlib.import_module(root_package), leaf_attr, None)
        assert restored_leaf is not None
        assert hasattr(restored_leaf, self.extender_name)

    def test_evicting_root_leaves_no_stale_module_behind(self) -> None:
        """A prior test using ``evict_root`` to poison a transitive dependency must not leak a stale
        ``root`` or transitive-dependency module into later tests."""
        if self.transitive_dependency is None:
            pytest.skip("no transitive_dependency declared")

        importlib.import_module(self.root)
        before_root = sys.modules[self.root]
        before_transitive = sys.modules.get(self.transitive_dependency)

        with pytest.MonkeyPatch.context() as mp:
            evict_root(mp, self.root)
            mp.setitem(sys.modules, self.transitive_dependency, None)
            evict_package(mp, self.package)

            with pytest.raises(ImportError):
                importlib.import_module(f"{self.package}.{self.extender_module}")

        assert sys.modules.get(self.root) is before_root
        assert sys.modules.get(self.transitive_dependency) is before_transitive

        extender_module = importlib.import_module(f"{self.package}.{self.extender_module}")
        assert hasattr(extender_module, self.extender_name)
