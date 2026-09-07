"""Contract for a community extender package whose runtime dependency is optional."""

from __future__ import annotations

import importlib
import logging
import sys

import pytest

from mloda.testing.import_isolation import block_root, evict_package


class OptionalDependencyPackageTestMixin:
    """Manifest resilience contract. Host declares package, root, distribution, extra, extender_name,
    extender_module: mloda's entry-point loader only tolerates a missing ``root`` (the manifest degrades
    to an empty EXTENDERS list), while any other import error must still propagate."""

    package: str
    root: str
    distribution: str
    extra: str
    extender_name: str
    extender_module: str

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
