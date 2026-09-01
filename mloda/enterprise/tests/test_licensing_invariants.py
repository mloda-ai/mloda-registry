"""Enterprise-wide licensing invariants: every enterprise plugin manifest must import with no
binary wheel installed, and every plugin using ``BinaryModelMixin`` must reject up front without a
license, never falling back to a Python computation (see
``docs/guides/feature-group-patterns/28-binary-backed-features.md``).
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, cast

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

import pyarrow as pa
import pytest

from mloda.community.feature_groups.binary_model.binary import clear_capability_cache
from mloda.community.feature_groups.binary_model.errors import BinaryUnavailableError, LicenseMissingError
from mloda.community.feature_groups.binary_model.mixin import BinaryModelMixin

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PACKAGES_CONFIG = _REPO_ROOT / "config" / "packages.toml"

# Entry-point group -> the manifest attribute it maps to (mirrors test_entry_points.py's _GROUP_INFO).
_GROUP_ATTR: dict[str, str] = {
    "mloda.feature_groups": "FEATURE_GROUPS",
    "mloda.compute_frameworks": "COMPUTE_FRAMEWORKS",
    "mloda.extenders": "EXTENDERS",
}

# The simulated binary from mloda-testing[binary-model]; its own plugin_id is "example_binary".
STUB_CMD = [sys.executable, "-m", "mloda.testing.binary_model.simulated_binary"]

_PLUGIN_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@pytest.fixture(autouse=True)
def _clear_capability_cache_before_each_test() -> None:
    clear_capability_cache()


def _load_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _enterprise_plugin_packages() -> list[tuple[str, dict[str, Any]]]:
    """Every enterprise package in config/packages.toml declaring entry_point_groups."""
    packages: dict[str, dict[str, Any]] = _load_toml(_PACKAGES_CONFIG).get("packages", {})
    return [
        (name, cfg)
        for name, cfg in packages.items()
        if cfg.get("path", "").startswith("mloda/enterprise/") and cfg.get("entry_point_groups")
    ]


def _plugin_classes_from_manifest(module: Any, cfg: dict[str, Any]) -> list[type[Any]]:
    classes: list[type[Any]] = []
    for group in cfg.get("entry_point_groups", []):
        attr_name = _GROUP_ATTR.get(group)
        if attr_name is None:
            continue
        classes.extend(getattr(module, attr_name, []))
    return classes


def _licensed_plugin_classes() -> list[type[BinaryModelMixin]]:
    """FEATURE_GROUPS classes across every enterprise manifest that subclass BinaryModelMixin."""
    licensed: list[type[BinaryModelMixin]] = []
    for _name, cfg in _enterprise_plugin_packages():
        dotted = cfg["path"].replace("/", ".")
        module = importlib.import_module(f"{dotted}.manifest")
        for cls in _plugin_classes_from_manifest(module, cfg):
            if issubclass(cls, BinaryModelMixin):
                licensed.append(cls)
    return licensed


class TestEnterprisePackageDiscovery:
    def test_binary_example_package_is_discovered(self) -> None:
        """Fails until config/packages.toml declares mloda-enterprise-binary-example, so this
        invariant suite can never silently skip the new plugin."""
        names = {name for name, _cfg in _enterprise_plugin_packages()}
        assert "mloda-enterprise-binary-example" in names


class TestEveryEnterpriseManifestImportsWithoutTheWheel:
    def test_every_manifest_module_imports(self) -> None:
        packages = _enterprise_plugin_packages()
        assert packages, "expected at least one enterprise plugin package"
        for _name, cfg in packages:
            dotted = cfg["path"].replace("/", ".")
            importlib.import_module(f"{dotted}.manifest")

    def test_no_binary_plugin_id_is_installed_as_a_wheel(self) -> None:
        licensed = _licensed_plugin_classes()
        assert licensed, "expected at least one licensed plugin class"
        for cls in licensed:
            assert importlib.util.find_spec(cls.BINARY_PLUGIN_ID) is None

    def test_importing_every_manifest_never_imports_a_binary_wheel(self) -> None:
        plugin_ids = {cls.BINARY_PLUGIN_ID for cls in _licensed_plugin_classes()}
        for _name, cfg in _enterprise_plugin_packages():
            dotted = cfg["path"].replace("/", ".")
            importlib.import_module(f"{dotted}.manifest")
        for plugin_id in plugin_ids:
            assert plugin_id not in sys.modules


class TestLicensedPluginsExist:
    def test_at_least_one_licensed_plugin_is_discovered(self) -> None:
        assert _licensed_plugin_classes()


class TestLicensedPluginsRejectWithoutLicense:
    def test_production_class_raises_binary_unavailable(self) -> None:
        table = pa.table({"col_a": ["alpha"]})
        licensed = _licensed_plugin_classes()
        assert licensed, "expected at least one licensed plugin class"
        for cls in licensed:
            with pytest.raises(BinaryUnavailableError):
                cls.run_binary_model(table, ["col_a"], "hash", {}, {"result": "out"})

    def test_stub_without_license_raises_license_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MLODA_LICENSE_FILE", raising=False)
        monkeypatch.delenv("MLODA_LICENSE_KEY", raising=False)
        table = pa.table({"col_a": ["alpha"]})
        licensed = _licensed_plugin_classes()
        assert licensed, "expected at least one licensed plugin class"
        for cls in licensed:
            stub_cls = cast(
                type[BinaryModelMixin],
                type(
                    f"Stub{cls.__name__}",
                    (cls,),
                    {
                        "BINARY_COMMAND_OVERRIDE": STUB_CMD,
                        "LICENSE_FILE_OVERRIDE": None,
                        "LICENSE_KEY_OVERRIDE": None,
                    },
                ),
            )
            operation = sorted(stub_cls.resolved_binary().capabilities.operations)[0]
            with pytest.raises(LicenseMissingError):
                stub_cls.run_binary_model(table, ["col_a"], operation, {}, {"result": "out"})


class TestBinaryPluginIdInvariants:
    def test_plugin_ids_are_unique(self) -> None:
        plugin_ids = [cls.BINARY_PLUGIN_ID for cls in _licensed_plugin_classes()]
        assert len(plugin_ids) == len(set(plugin_ids))

    def test_plugin_ids_match_naming_convention(self) -> None:
        licensed = _licensed_plugin_classes()
        assert licensed, "expected at least one licensed plugin class"
        for cls in licensed:
            assert _PLUGIN_ID_PATTERN.match(cls.BINARY_PLUGIN_ID), cls.BINARY_PLUGIN_ID
