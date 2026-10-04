"""Wheel-to-package binding in scripts/verify_builds.py. Every package builds into one shared temp
directory, so a wheel must be bound by its distribution name alone: neither config order nor the
version setuptools normalizes into the filename may decide which wheel a package is verified against."""

from __future__ import annotations

import shutil
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.script_loader import load_script

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VERIFY_BUILDS_PATH = _REPO_ROOT / "scripts" / "verify_builds.py"
_PUBLISHED_PACKAGES_PATH = _REPO_ROOT / "scripts" / "published_packages.py"

_VERSION = "0.4.0"

# A non-canonical PEP 440 spelling of _VERSION; setuptools normalizes it away in the wheel filename.
_NON_CANONICAL_VERSION = "0.04.0"

# Version of a same-distribution wheel left behind in a reused out-dir by an earlier run.
_STALE_VERSION = "0.3.9"

# Children before the bundle whose name is their prefix, the inverse of config/packages.toml today.
_REORDERED_NAMES = ["mloda-community-example", "mloda-community-offset", "mloda-community", "mloda-registry"]

vb = load_script("verify_builds", _VERIFY_BUILDS_PATH)
pp = load_script("published_packages", _PUBLISHED_PACKAGES_PATH)


def _find_wheels() -> Callable[[Path, str], list[Path]]:
    """The name-only wheel lookup verify_builds must expose."""
    finder: Callable[[Path, str], list[Path]] | None = getattr(vb, "find_wheels", None)
    assert callable(finder), "verify_builds.find_wheels(out_dir, pkg_name) must be a callable"
    return finder


def _escape_distribution_name() -> Callable[[str], str]:
    """The distribution-name escaping verify_builds must expose."""
    escape: Callable[[str], str] | None = getattr(vb, "escape_distribution_name", None)
    assert callable(escape), "verify_builds.escape_distribution_name(name) must be a callable"
    return escape


class _FakeCompletedProcess:
    """Stand-in for subprocess.CompletedProcess; main() reads returncode and stderr."""

    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.stdout = ""
        self.stderr = "stubbed: the pytest gate never runs a real build"


def _wheel_name(pkg_name: str, version: str = _VERSION) -> str:
    """Wheel filename a real build produces; configured names carry only '-', so this replace suffices."""
    return f"{pkg_name.replace('-', '_')}-{version}-py3-none-any.whl"


def _write_wheel(out_dir: Path, pkg_name: str, version: str = _VERSION) -> Path:
    """A minimal real wheel: a zip whose dist-info METADATA declares ``version``."""
    path = out_dir / _wheel_name(pkg_name, version)
    dist_info = f"{pkg_name.replace('-', '_')}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{dist_info}/METADATA", f"Metadata-Version: 2.4\nName: {pkg_name}\nVersion: {version}\n")
    return path


def _write_wheel_with_files(out_dir: Path, pkg_name: str, files: list[str], version: str = _VERSION) -> Path:
    """Like ``_write_wheel``, but the zip also carries the given (already wheel-relative) file paths."""
    path = out_dir / _wheel_name(pkg_name, version)
    dist_info = f"{pkg_name.replace('-', '_')}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{dist_info}/METADATA", f"Metadata-Version: 2.4\nName: {pkg_name}\nVersion: {version}\n")
        for file_path in files:
            zf.writestr(file_path, "")
    return path


def _write_wheel_with_metadata(out_dir: Path, pkg_name: str, extra_lines: list[str], version: str = _VERSION) -> Path:
    """Like ``_write_wheel``, but the METADATA carries the given additional lines (extras, requires-dist)."""
    path = out_dir / _wheel_name(pkg_name, version)
    dist_info = f"{pkg_name.replace('-', '_')}-{version}.dist-info"
    lines = ["Metadata-Version: 2.4", f"Name: {pkg_name}", f"Version: {version}", *extra_lines]
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{dist_info}/METADATA", "\n".join(lines) + "\n")
    return path


def _sandbox(root: Path, monkeypatch: pytest.MonkeyPatch, names: list[str]) -> list[tuple[str, str]]:
    """Run main() against a copy of the config; returns the (name, pyproject path) entries for ``names``."""
    (root / "config").mkdir(parents=True, exist_ok=True)
    shutil.copy(_REPO_ROOT / "config" / "packages.toml", root / "config" / "packages.toml")
    monkeypatch.chdir(root)

    paths = dict(vb.load_packages_from_config())
    missing = [name for name in names if name not in paths]
    assert not missing, f"fixture assumption: config/packages.toml must declare {missing}"
    return [(name, paths[name]) for name in names]


def _stub_build(
    monkeypatch: pytest.MonkeyPatch,
    packages: list[tuple[str, str]],
    *,
    declared_version: str = _VERSION,
    skip_wheel: str | None = None,
    stale_wheels: dict[str, str] | None = None,
) -> dict[str, Path]:
    """Drive main() with fake builds; returns the built_wheels mapping, filled in once main() runs.

    ``declared_version`` is what the configs claim. The wheels always carry _VERSION, because setuptools
    writes the normalized version into both the filename and METADATA. verify_wheel_version stays real.
    """
    captured: dict[str, Path] = {}

    def _fake_build(cmd: list[str], *args: Any, **kwargs: Any) -> _FakeCompletedProcess:
        out_dir = Path(cmd[cmd.index("--out-dir") + 1])
        package = cmd[cmd.index("--package") + 1]
        if package != skip_wheel:
            _write_wheel(out_dir, package)
        stale = (stale_wheels or {}).get(package)
        if stale is not None:
            _write_wheel(out_dir, package, stale)
        return _FakeCompletedProcess()

    def _consistent_versions() -> tuple[bool, str]:
        return True, declared_version

    def _capture_wheels(built_wheels: dict[str, Path]) -> list[str]:
        captured.update(built_wheels)
        return []

    def _no_errors(*args: Any, **kwargs: Any) -> list[str]:
        return []

    monkeypatch.setattr(vb, "PACKAGES", packages)
    monkeypatch.setattr(vb.subprocess, "run", _fake_build)
    monkeypatch.setattr(vb, "check_version_consistency", _consistent_versions)
    monkeypatch.setattr(vb, "verify_entry_points", _capture_wheels)
    for verifier in (
        "verify_dependency_relationships",
        "verify_wheel_metadata",
        "verify_py_typed_markers",
        "verify_pep420_source_compliance",
    ):
        monkeypatch.setattr(vb, verifier, _no_errors)
    return captured


def test_a_prefix_sibling_wheel_never_matches_the_shorter_name(tmp_path: Path) -> None:
    """Order independent by construction: the distribution segment is compared whole, not as a prefix."""
    own = _write_wheel(tmp_path, "mloda-community")
    sibling = _write_wheel(tmp_path, "mloda-community-offset")
    find_wheels = _find_wheels()

    matched = find_wheels(tmp_path, "mloda-community")

    assert sibling not in matched, f"find_wheels matched the prefix sibling {sibling.name} for mloda-community"
    assert matched == [own], f"mloda-community must match only {own.name}, got {[path.name for path in matched]}"
    assert find_wheels(tmp_path, "mloda-community-offset") == [sibling], (
        f"mloda-community-offset must match only {sibling.name}"
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("mloda-community", "mloda_community"),
        ("mloda-community-io.parquet", "mloda_community_io_parquet"),
        ("mloda--community", "mloda_community"),
        ("mloda-_.community", "mloda_community"),
        ("Mloda-Community", "mloda_community"),
        ("mloda_community", "mloda_community"),
    ],
)
def test_distribution_name_escaping_follows_the_wheel_spec(raw: str, expected: str) -> None:
    """PEP 427/503 escaping is re.sub(r"[-_.]+", "_", name.lower()), not a bare '-' to '_' replace."""
    escape = _escape_distribution_name()
    assert escape(raw) == expected, f"escaping {raw!r} produced {escape(raw)!r}, expected {expected!r}"


def test_every_package_binds_its_own_wheel_under_a_reordered_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config order must not decide which wheel a package is verified against."""
    packages = _sandbox(tmp_path, monkeypatch, _REORDERED_NAMES)
    built_wheels = _stub_build(monkeypatch, packages)

    exit_code = vb.main()

    assert exit_code == 0, "fixture assumption: the stubs must drive main() down its success path"
    assert sorted(built_wheels) == sorted(_REORDERED_NAMES), (
        f"main() bound wheels for {sorted(built_wheels)}, expected {sorted(_REORDERED_NAMES)}"
    )
    for pkg_name, wheel_path in built_wheels.items():
        assert wheel_path.name == _wheel_name(pkg_name), (
            f"{pkg_name}: bound {wheel_path.name}, a wheel carrying another distribution's name; "
            f"expected {_wheel_name(pkg_name)}"
        )


def test_a_prefix_sibling_wheel_is_never_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With its own wheel absent, a package must report no wheel instead of adopting a sibling's."""
    packages = _sandbox(tmp_path, monkeypatch, _REORDERED_NAMES)
    built_wheels = _stub_build(monkeypatch, packages, skip_wheel="mloda-community")

    exit_code = vb.main()
    output = capsys.readouterr().out

    assert "mloda-community" not in built_wheels, (
        f"main() bound {built_wheels.get('mloda-community')} to mloda-community, which built no wheel"
    )
    assert "mloda-community: no wheel produced" in output, (
        f"main() must report the missing mloda-community wheel, printed:\n{output}"
    )
    assert exit_code == 1, f"main() must fail when a package produces no wheel, returned {exit_code!r}"


def test_a_normalized_wheel_version_is_diagnosed_as_a_version_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-canonical configured version is normalized out of the filename; the wheel is still bound."""
    packages = _sandbox(tmp_path, monkeypatch, ["mloda-registry"])
    built_wheels = _stub_build(monkeypatch, packages, declared_version=_NON_CANONICAL_VERSION)

    exit_code = vb.main()
    output = capsys.readouterr().out

    assert "no wheel produced" not in output, (
        f"main() lost {_wheel_name('mloda-registry')} because the configs declare {_NON_CANONICAL_VERSION}, "
        f"which setuptools normalized to {_VERSION} in the filename; printed:\n{output}"
    )
    assert "mloda-registry: version mismatch in wheel" in output, (
        f"main() must diagnose the wheel it found as a version mismatch, printed:\n{output}"
    )
    assert "mloda-registry" not in built_wheels, "a wheel failing version verification must not be verified further"
    assert exit_code == 1, f"main() must fail on a version mismatch, returned {exit_code!r}"


def _verify_published_wheels_have_a_single_owner() -> Callable[[dict[str, Path], list[str]], list[str]]:
    """The published-pair overlap check verify_builds must expose, replacing verify_shared_wheel_has_single_owner."""
    verify: Callable[[dict[str, Path], list[str]], list[str]] | None = getattr(
        vb, "verify_published_wheels_have_a_single_owner", None
    )
    assert callable(verify), "verify_builds.verify_published_wheels_have_a_single_owner must be a callable"
    return verify


def test_overlap_between_two_published_wheels_is_reported(tmp_path: Path) -> None:
    """A path shipped by two published wheels' file lists is a single-owner violation."""
    verify = _verify_published_wheels_have_a_single_owner()
    shared_path = "mloda/community/extenders/shared/foo.py"
    bundle = _write_wheel_with_files(tmp_path, "mloda-community", ["mloda/community/py.typed", shared_path])
    shared = _write_wheel_with_files(tmp_path, "mloda-community-extenders-shared", [shared_path])
    wheels = {"mloda-community": bundle, "mloda-community-extenders-shared": shared}

    errors = verify(wheels, ["mloda-community", "mloda-community-extenders-shared"])

    assert any(shared_path in error for error in errors), (
        f"expected an overlap error naming {shared_path!r}, got {errors!r}"
    )


def test_overlap_check_ignores_dist_info_entries(tmp_path: Path) -> None:
    """Every wheel's own dist-info carries files (RECORD, METADATA, ...); those must never count as overlap."""
    verify = _verify_published_wheels_have_a_single_owner()
    # Same literal dist-info-relative path in both wheels, to isolate the ignore rule from real dist-info naming.
    shared_dist_info_path = "shared.dist-info/RECORD"
    one = _write_wheel_with_files(tmp_path, "mloda-registry", [shared_dist_info_path])
    other = _write_wheel_with_files(tmp_path, "mloda-testing", [shared_dist_info_path])
    wheels = {"mloda-registry": one, "mloda-testing": other}

    errors = verify(wheels, ["mloda-registry", "mloda-testing"])

    assert errors == [], f"a shared *.dist-info/ entry must never be reported as an overlap, got {errors!r}"


def test_overlap_check_ignores_an_unpublished_wheel(tmp_path: Path) -> None:
    """An overlap that involves a wheel outside the published set must not be reported."""
    verify = _verify_published_wheels_have_a_single_owner()
    shared_path = "mloda/community/extenders/shared/foo.py"
    bundle = _write_wheel_with_files(tmp_path, "mloda-community", [shared_path])
    unpublished = _write_wheel_with_files(tmp_path, "mloda-community-binary-model", [shared_path])
    wheels = {"mloda-community": bundle, "mloda-community-binary-model": unpublished}

    errors = verify(wheels, ["mloda-community"])

    assert errors == [], (
        f"an overlap with an unpublished wheel must be ignored, {shared_path!r} must not be reported: {errors!r}"
    )


def test_two_wheels_for_one_distribution_are_rejected_as_ambiguous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Name-only matching can also find a stale wheel in a reused out-dir; picking one silently is worse."""
    packages = _sandbox(tmp_path, monkeypatch, ["mloda-registry"])
    built_wheels = _stub_build(monkeypatch, packages, stale_wheels={"mloda-registry": _STALE_VERSION})

    exit_code = vb.main()
    output = capsys.readouterr().out

    assert "mloda-registry" not in built_wheels, (
        f"main() silently bound {built_wheels.get('mloda-registry')} out of two candidate wheels"
    )
    for candidate in (_wheel_name("mloda-registry"), _wheel_name("mloda-registry", _STALE_VERSION)):
        assert candidate in output, f"main() must name the ambiguous candidate {candidate}, printed:\n{output}"
    assert exit_code == 1, f"main() must fail when one distribution has two wheels, returned {exit_code!r}"


def test_community_example_wheel_with_example_a_in_the_all_extra_reports_no_error(tmp_path: Path) -> None:
    """A community example wheel whose 'all' extra lists example-a is the happy path for
    verify_dependency_relationships."""
    wheel = _write_wheel_with_metadata(
        tmp_path,
        "mloda-community-example",
        ["Provides-Extra: all", 'Requires-Dist: mloda-community-example-a; extra == "all"'],
    )

    errors = vb.verify_dependency_relationships({"mloda-community-example": wheel})

    assert errors == [], f"a wheel whose 'all' extra lists only example-a must report no error, got {errors!r}"


def _published_wheels() -> Callable[[dict[str, dict[str, Any]], Path], list[Path]]:
    """The one-wheel-per-published-distribution matcher published_packages.py must expose."""
    fn: Callable[[dict[str, dict[str, Any]], Path], list[Path]] | None = getattr(pp, "published_wheels", None)
    assert callable(fn), "published_packages.published_wheels(packages, out_dir) must be a callable"
    return fn


def test_published_wheels_follows_published_order_not_filename_order(tmp_path: Path) -> None:
    """The returned list order is the published (config) order, not the filename/glob order on disk."""
    packages: dict[str, dict[str, Any]] = {
        "mloda-testing": {"description": "sandbox", "path": "mloda/testing", "published": True},
        "mloda-registry": {"description": "sandbox", "path": "mloda/registry", "published": True},
    }
    # Filenames sort mloda-registry before mloda-testing, the reverse of the published (config) order above.
    _write_wheel(tmp_path, "mloda-registry")
    _write_wheel(tmp_path, "mloda-testing")

    wheels = _published_wheels()(packages, tmp_path)

    expected = [_wheel_name("mloda-testing"), _wheel_name("mloda-registry")]
    assert [w.name for w in wheels] == expected, (
        f"published_wheels() returned {[w.name for w in wheels]!r}, expected published order {expected!r}"
    )


def test_published_wheels_never_matches_a_prefix_sibling(tmp_path: Path) -> None:
    """A prefix sibling on disk (e.g. mloda-community-offset) must never satisfy mloda-community."""
    packages: dict[str, dict[str, Any]] = {
        "mloda-community": {"description": "sandbox", "path": "mloda/community", "published": True},
    }
    own = _write_wheel(tmp_path, "mloda-community")
    sibling = _write_wheel(tmp_path, "mloda-community-offset")

    wheels = _published_wheels()(packages, tmp_path)

    assert sibling not in wheels, f"published_wheels() matched the prefix sibling {sibling.name}"
    assert wheels == [own], f"published_wheels() must match only {own.name}, got {[w.name for w in wheels]}"


def test_published_wheels_raises_when_no_wheel_matches(tmp_path: Path) -> None:
    """Zero matching wheels for a published package must raise, naming that package."""
    packages: dict[str, dict[str, Any]] = {
        "mloda-registry": {"description": "sandbox", "path": "mloda/registry", "published": True},
    }

    with pytest.raises(ValueError, match="mloda-registry"):
        _published_wheels()(packages, tmp_path)


def test_published_wheels_raises_when_two_wheels_match(tmp_path: Path) -> None:
    """Two matching wheels for one published package (e.g. a stale wheel left in a reused out-dir) must
    raise, naming that package, rather than silently picking one."""
    packages: dict[str, dict[str, Any]] = {
        "mloda-registry": {"description": "sandbox", "path": "mloda/registry", "published": True},
    }
    _write_wheel(tmp_path, "mloda-registry", _VERSION)
    _write_wheel(tmp_path, "mloda-registry", _STALE_VERSION)

    with pytest.raises(ValueError, match="mloda-registry"):
        _published_wheels()(packages, tmp_path)


def _write_packages_config(root: Path, body: str) -> None:
    """A minimal config/packages.toml under ``root``, for driving published_packages.py's CLI."""
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "config" / "packages.toml").write_text(body)


def test_cli_wheels_prints_matched_wheel_paths_one_per_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--wheels DIR`` prints the matched wheel paths, one per line, in published order."""
    _write_packages_config(
        tmp_path,
        '[packages.mloda-registry]\ndescription = "sandbox"\npath = "mloda/registry"\npublished = true\n',
    )
    wheel = _write_wheel(tmp_path, "mloda-registry")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["published_packages.py", "--wheels", str(tmp_path)])

    main: Callable[[], int] | None = getattr(pp, "main", None)
    assert callable(main), "published_packages.main must be a callable returning an exit code"
    exit_code = main()

    out = capsys.readouterr().out
    assert exit_code == 0, f"'published_packages.py --wheels {tmp_path}' exited {exit_code!r}, expected 0"
    assert out.strip().splitlines() == [str(wheel)], (
        f"'published_packages.py --wheels {tmp_path}' printed {out!r}, expected exactly the matched wheel path"
    )


def test_cli_wheels_exits_non_zero_when_published_wheels_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A published_wheels() ValueError (no wheel found here) must become a non-zero exit with the reason on
    stderr, not an uncaught traceback."""
    _write_packages_config(
        tmp_path,
        '[packages.mloda-registry]\ndescription = "sandbox"\npath = "mloda/registry"\npublished = true\n',
    )
    # No wheel written: published_wheels() must raise, naming mloda-registry.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["published_packages.py", "--wheels", str(tmp_path)])

    main: Callable[[], int] | None = getattr(pp, "main", None)
    assert callable(main), "published_packages.main must be a callable returning an exit code"
    try:
        exit_code = main()
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 1

    err = capsys.readouterr().err
    assert exit_code != 0, "published_packages.main() must exit non-zero when published_wheels() raises"
    assert "mloda-registry" in err, f"stderr must name the offending package, got {err!r}"


def _write_misordered_packages_config(root: Path) -> None:
    """A published package naming a later-declared published sibling, both modes must report this cleanly."""
    _write_packages_config(
        root,
        '[packages.pkg-a]\ndescription = "sandbox"\npath = "p/a"\npublished = true\n'
        'dependencies = ["pkg-b>=1.0"]\n\n'
        '[packages.pkg-b]\ndescription = "sandbox"\npath = "p/b"\npublished = true\n',
    )


@pytest.mark.parametrize("wheels_mode", [False, True], ids=["bare", "wheels"])
def test_cli_reports_a_misordered_published_config_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], wheels_mode: bool
) -> None:
    """published_packages() raises ValueError for a mis-ordered config; main() must turn that into a clean
    non-zero exit with the reason (naming both packages) on stderr, not an uncaught traceback, in both the
    plain mode and '--wheels' mode."""
    _write_misordered_packages_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    argv = ["published_packages.py", "--wheels", str(tmp_path)] if wheels_mode else ["published_packages.py"]
    monkeypatch.setattr(sys, "argv", argv)

    main: Callable[[], int] | None = getattr(pp, "main", None)
    assert callable(main), "published_packages.main must be a callable returning an exit code"
    try:
        exit_code = main()
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 1

    err = capsys.readouterr().err
    mode = "--wheels" if wheels_mode else "bare"
    assert exit_code != 0, (
        f"published_packages.main() ({mode} mode) must exit non-zero for a mis-ordered published config, "
        "not raise an uncaught ValueError"
    )
    assert "pkg-a" in err and "pkg-b" in err, (
        f"published_packages.main() ({mode} mode) must name both pkg-a and pkg-b on stderr, got {err!r}"
    )


def test_cli_wheels_exits_non_zero_when_nothing_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty published set would hand the release workflow an empty upload list; --wheels must fail
    loudly too, not just the bare invocation."""
    _write_packages_config(
        tmp_path,
        '[packages.mloda-registry]\ndescription = "sandbox"\npath = "mloda/registry"\n',
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["published_packages.py", "--wheels", str(tmp_path)])

    main: Callable[[], int] | None = getattr(pp, "main", None)
    assert callable(main), "published_packages.main must be a callable returning an exit code"
    try:
        exit_code = main()
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 1

    err = capsys.readouterr().err
    assert exit_code != 0, (
        "published_packages.main() must exit non-zero for '--wheels' when no package is flagged "
        "'published = true', the same as the bare invocation"
    )
    assert err.strip(), "the reason must be printed on stderr, not silently swallowed"
