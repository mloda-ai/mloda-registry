"""Every package in the committed ``uv.lock`` is a workspace member or resolves from production PyPI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

from mloda.enterprise.feature_groups.binary_example.binary_example_feature_group import BinaryExampleFeatureGroup
from tests.script_loader import load_script

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCK_PATH = _REPO_ROOT / "uv.lock"

gen = load_script("generate_pyproject", _REPO_ROOT / "scripts" / "generate_pyproject.py")

_PYPI_REGISTRY = "https://pypi.org/simple"
_PYPI_FILES_HOST = "https://files.pythonhosted.org/"

_DIGEST = "sha256:" + "0" * 64
_WHEEL_URL = _PYPI_FILES_HOST + "packages/aa/bb/dep-1.0.0-py3-none-any.whl"
_SDIST_URL = _PYPI_FILES_HOST + "packages/aa/bb/dep-1.0.0.tar.gz"


def _normalised(name: str) -> str:
    """PEP 503 normalised distribution name, or the name unchanged when it has none."""
    return gen.normalize_dependency_name(name) or name


def _real_lock() -> dict[str, Any]:
    with open(_LOCK_PATH, "rb") as f:
        return tomllib.load(f)


def _sources(lock: dict[str, Any]) -> list[dict[str, Any]]:
    """The ``source`` table of every locked package that has one."""
    return [package["source"] for package in lock.get("package", []) if isinstance(package.get("source"), dict)]


def _is_pypi_source(source: Any) -> bool:
    return isinstance(source, dict) and str(source.get("registry", "")).rstrip("/") == _PYPI_REGISTRY


def _artifact_violations(name: str, package: dict[str, Any]) -> list[str]:
    """Violations among the wheels and sdist of a registry package: off the files host, or not sha256-hashed."""
    artifacts: list[tuple[str, dict[str, Any]]] = [("wheel", wheel) for wheel in package.get("wheels", [])]
    if "sdist" in package:
        artifacts.append(("sdist", package["sdist"]))
    violations: list[str] = []
    for kind, artifact in artifacts:
        url = str(artifact.get("url", ""))
        if not url.startswith(_PYPI_FILES_HOST):
            violations.append(f"{name}: {kind} url {url!r} is not under {_PYPI_FILES_HOST}")
        if not str(artifact.get("hash", "")).startswith("sha256:"):
            violations.append(f"{name}: {kind} {url!r} has no sha256 hash")
    return violations


def _lock_violations(lock: dict[str, Any]) -> list[str]:
    """Each package of ``lock`` outside the source allow-list, named; an empty list means compliant.

    Allowed: workspace members (``editable`` or ``virtual``) and ``registry`` packages from production PyPI,
    whose wheels and sdist must come from the PyPI files host and carry a sha256 hash.
    """
    violations: list[str] = []
    for package in lock.get("package", []):
        name = package.get("name", "<unnamed>")
        source = package.get("source")
        if not isinstance(source, dict):
            violations.append(f"{name}: no source")
        elif set(source) in ({"editable"}, {"virtual"}):
            continue
        elif set(source) != {"registry"} or not _is_pypi_source(source):
            violations.append(f"{name}: source {source!r} is not a workspace member or {_PYPI_REGISTRY}")
        else:
            violations.extend(_artifact_violations(name, package))
    return violations


def test_real_lock_sources_are_workspace_or_production_pypi() -> None:
    """Every locked package is a workspace member or comes, hashed, from production PyPI."""
    lock = _real_lock()
    violations = _lock_violations(lock)
    assert violations == [], f"uv.lock has {len(violations)} source violation(s):\n" + "\n".join(violations)

    sources = _sources(lock)
    registries = [str(source["registry"]) for source in sources if "registry" in source]
    assert registries, "uv.lock records no registry-sourced packages, so the source check would be vacuous"
    assert any(url.rstrip("/") == _PYPI_REGISTRY for url in registries), (
        f"uv.lock records no package from {_PYPI_REGISTRY}, so the source check would be vacuous; "
        f"registries seen: {sorted(set(registries))!r}"
    )
    assert any("editable" in source or "virtual" in source for source in sources), (
        "uv.lock records no workspace member (editable or virtual source), so the source check would be vacuous"
    )


def test_binary_wheel_is_locked_from_production_pypi() -> None:
    """The wheel behind the binary-backed extra is in the lock and its source is production PyPI."""
    wheel = _normalised(BinaryExampleFeatureGroup.BINARY_WHEEL_DISTRIBUTION)
    packages = _real_lock().get("package", [])
    locked = [package for package in packages if _normalised(str(package["name"])) == wheel]
    assert locked, f"uv.lock must contain {wheel!r}, the wheel behind the binary-backed extra; {len(packages)} locked"
    offenders = [
        f"{package['name']} ({package.get('source')!r})"
        for package in locked
        if not _is_pypi_source(package.get("source"))
    ]
    assert offenders == [], f"{wheel!r} must be locked from {_PYPI_REGISTRY}, found: {', '.join(offenders)}"


def _lock(*packages: dict[str, Any]) -> dict[str, Any]:
    return {"version": 1, "package": list(packages)}


def _artifact(url: str, digest: str | None = _DIGEST) -> dict[str, Any]:
    """A wheel or sdist entry; ``digest=None`` leaves the ``hash`` key out."""
    artifact: dict[str, Any] = {"url": url, "size": 1}
    if digest is not None:
        artifact["hash"] = digest
    return artifact


def _local_package(name: str, source: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "version": "0.1.0", "source": source}


def _registry_package(
    name: str,
    registry: str = _PYPI_REGISTRY,
    wheels: list[dict[str, Any]] | None = None,
    sdist: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A registry package; compliant unless ``registry``, ``wheels`` or ``sdist`` are overridden."""
    return {
        "name": name,
        "version": "1.0.0",
        "source": {"registry": registry},
        "sdist": sdist if sdist is not None else _artifact(_SDIST_URL),
        "wheels": wheels if wheels is not None else [_artifact(_WHEEL_URL)],
    }


def _assert_only_offender_named(violations: list[str], offender: str) -> None:
    assert violations, f"expected a violation naming {offender!r}, got none"
    assert all(offender in violation for violation in violations), (
        f"every violation must name {offender!r} and none may name a compliant package: {violations!r}"
    )


_REJECTED_SOURCES: dict[str, dict[str, Any]] = {
    "git": {"git": "https://github.com/example/bad-dep?rev=0123456#0123456"},
    "url": {"url": "https://example.org/bad-dep-1.0.0.tar.gz"},
    "path": {"path": "../bad-dep"},
    "directory": {"directory": "vendor/bad-dep"},
    "testpypi": {"registry": "https://test.pypi.org/simple/"},
    "corporate_mirror": {"registry": "https://pypi.corp.example/simple"},
    "http_pypi": {"registry": "http://pypi.org/simple"},
    "lookalike_subdomain": {"registry": "https://pypi.org.evil.example/simple"},
    "lookalike_path": {"registry": "https://evil.example/pypi.org/simple"},
    "uppercase_host": {"registry": "https://PyPI.org/simple"},
}


class TestLockViolations:
    """Exercises ``_lock_violations`` against synthetic lock data, so a regression is caught even
    when the real ``uv.lock`` is already compliant."""

    def test_accepts_workspace_members_and_pypi_packages(self) -> None:
        lock = _lock(
            _local_package("member-editable", {"editable": "mloda/community/example"}),
            _local_package("member-virtual", {"virtual": "."}),
            _registry_package("good-dep"),
        )
        assert _lock_violations(lock) == []

    def test_accepts_a_trailing_slash_registry_url(self) -> None:
        lock = _lock(_registry_package("good-dep", registry="https://pypi.org/simple/"))
        assert _lock_violations(lock) == []

    def test_accepts_wheel_only_and_sdist_only_registry_packages(self) -> None:
        wheel_only = _registry_package("wheel-only")
        del wheel_only["sdist"]
        sdist_only = _registry_package("sdist-only")
        del sdist_only["wheels"]
        assert _lock_violations(_lock(wheel_only, sdist_only)) == []

    @pytest.mark.parametrize("source", list(_REJECTED_SOURCES.values()), ids=list(_REJECTED_SOURCES))
    def test_rejects_a_source_outside_the_allow_list(self, source: dict[str, Any]) -> None:
        lock = _lock(_registry_package("good-dep"), _local_package("bad-dep", source))
        _assert_only_offender_named(_lock_violations(lock), "bad-dep")

    def test_rejects_a_package_with_no_source(self) -> None:
        lock = _lock(_registry_package("good-dep"), {"name": "bad-dep", "version": "1.0.0"})
        _assert_only_offender_named(_lock_violations(lock), "bad-dep")

    @pytest.mark.parametrize(
        "url",
        [
            "https://evil.example/packages/aa/bb/bad_dep-1.0.0-py3-none-any.whl",
            "https://files.pythonhosted.org.evil.example/packages/aa/bb/bad_dep-1.0.0-py3-none-any.whl",
            "http://files.pythonhosted.org/packages/aa/bb/bad_dep-1.0.0-py3-none-any.whl",
        ],
        ids=["foreign_host", "lookalike_host", "http_scheme"],
    )
    def test_rejects_a_pypi_package_with_a_wheel_url_off_the_files_host(self, url: str) -> None:
        bad = _registry_package("bad-dep", wheels=[_artifact(_WHEEL_URL), _artifact(url)])
        _assert_only_offender_named(_lock_violations(_lock(_registry_package("good-dep"), bad)), "bad-dep")

    def test_rejects_a_pypi_package_with_an_sdist_url_off_the_files_host(self) -> None:
        bad = _registry_package("bad-dep", sdist=_artifact("https://evil.example/packages/aa/bb/bad_dep-1.0.0.tar.gz"))
        _assert_only_offender_named(_lock_violations(_lock(_registry_package("good-dep"), bad)), "bad-dep")

    def test_rejects_a_wheel_with_no_hash(self) -> None:
        bad = _registry_package("bad-dep", wheels=[_artifact(_WHEEL_URL), _artifact(_WHEEL_URL, digest=None)])
        _assert_only_offender_named(_lock_violations(_lock(_registry_package("good-dep"), bad)), "bad-dep")

    def test_rejects_a_wheel_with_a_non_sha256_hash(self) -> None:
        bad = _registry_package(
            "bad-dep", wheels=[_artifact(_WHEEL_URL), _artifact(_WHEEL_URL, digest="md5:" + "0" * 32)]
        )
        _assert_only_offender_named(_lock_violations(_lock(_registry_package("good-dep"), bad)), "bad-dep")

    def test_rejects_an_sdist_with_no_hash(self) -> None:
        bad = _registry_package("bad-dep", sdist=_artifact(_SDIST_URL, digest=None))
        _assert_only_offender_named(_lock_violations(_lock(_registry_package("good-dep"), bad)), "bad-dep")
