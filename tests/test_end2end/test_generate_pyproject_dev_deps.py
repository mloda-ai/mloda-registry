"""Tests pinning down the root dev-extras auto-sync feature for scripts/generate_pyproject.py.

A package needing a third-party dependency only for its own dev/test extra (e.g.
``mloda-community-otel``'s ``opentelemetry-sdk``) also needs a matching entry in root
``pyproject.toml``'s ``[project.optional-dependencies].dev``, since tox's shared dev
environment installs from the root file rather than each workspace member's own extras.
This file pins the contract for deriving those root entries from ``config/packages.toml``
automatically, the same way ``update_workspace_members`` and ``update_root_core_dependency``
already auto-derive workspace members and the core pin (see ``test_generate_pyproject_guards.py``).

The generator lives at ``scripts/generate_pyproject.py`` (a script, not an installed
package), so it is loaded here by file path.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from tests.script_loader import load_script

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GEN_PATH = _REPO_ROOT / "scripts" / "generate_pyproject.py"
_ROOT_PYPROJECT = _REPO_ROOT / "pyproject.toml"

gen = load_script("generate_pyproject", _GEN_PATH)


def _dev_array_block(entries: list[str]) -> str:
    lines = ["dev = ["]
    lines.extend(f'    "{entry}",' for entry in entries)
    lines.append("]")
    return "\n".join(lines)


def _sandbox_root_pyproject(dev_entries: list[str]) -> str:
    """Minimal but realistic root pyproject.toml with a controlled hand-authored dev list."""
    return (
        "[project]\n"
        'name = "sandbox-root"\n'
        "dependencies = [\n"
        '    "mloda>=0.12.0,<0.13.0",\n'
        "]\n"
        "\n"
        "[project.optional-dependencies]\n"
        f"{_dev_array_block(dev_entries)}\n"
    )


def _dev_list(pyproject_text: str) -> list[str]:
    """Parse the ``[project.optional-dependencies].dev`` array out of pyproject TOML text."""
    data = gen.tomllib.loads(pyproject_text)
    return list(data["project"]["optional-dependencies"]["dev"])


def _copy_real_configs(dest: Path) -> None:
    (dest / "config").mkdir(parents=True, exist_ok=True)
    shutil.copy(_REPO_ROOT / "config" / "shared.toml", dest / "config" / "shared.toml")
    shutil.copy(_REPO_ROOT / "config" / "packages.toml", dest / "config" / "packages.toml")


# --- collect_third_party_dev_deps ------------------------------------------------------------


def test_collect_third_party_dev_deps_excludes_internal_packages() -> None:
    """An internal workspace package named in a dev extra (e.g. mloda-testing) is never synced."""
    packages = {
        "mloda-testing": {"path": "mloda/testing", "dependencies": []},
        "mloda-community-otel": {
            "path": "mloda/community/extenders/otel",
            "dependencies": [],
            "optional_dependencies": {"dev": ["mloda-testing", "opentelemetry-sdk>=1.30,<2"]},
        },
    }

    result = gen.collect_third_party_dev_deps(packages)

    assert result == [("opentelemetry-sdk>=1.30,<2", "mloda-community-otel")]


def test_collect_third_party_dev_deps_normalizes_internal_name_match() -> None:
    """Internal-name matching is case-insensitive with ``_``/``.`` treated the same as ``-``."""
    packages = {
        "mloda-testing": {"path": "mloda/testing", "dependencies": []},
        "mloda-community-otel": {
            "path": "mloda/community/extenders/otel",
            "dependencies": [],
            "optional_dependencies": {"dev": ["Mloda_Testing>=1.0", "opentelemetry-sdk>=1.30,<2"]},
        },
    }

    result = gen.collect_third_party_dev_deps(packages)

    assert result == [("opentelemetry-sdk>=1.30,<2", "mloda-community-otel")]


def test_collect_third_party_dev_deps_dedupes_matching_requirement_across_packages() -> None:
    """The same third-party requirement declared by two packages collapses to one entry."""
    packages = {
        "pkg-a": {"path": "mloda/a", "dependencies": [], "optional_dependencies": {"dev": ["pytest>=9.0.3"]}},
        "pkg-b": {"path": "mloda/b", "dependencies": [], "optional_dependencies": {"dev": ["pytest>=9.0.3"]}},
    }

    result = gen.collect_third_party_dev_deps(packages)

    assert result == [("pytest>=9.0.3", "pkg-a")]


def test_collect_third_party_dev_deps_raises_on_conflicting_specs() -> None:
    """Two packages pinning different requirement strings for the same bare name must fail loudly."""
    packages = {
        "pkg-a": {"path": "mloda/a", "dependencies": [], "optional_dependencies": {"dev": ["foo>=1.0"]}},
        "pkg-b": {"path": "mloda/b", "dependencies": [], "optional_dependencies": {"dev": ["foo>=2.0"]}},
    }

    with pytest.raises(ValueError, match="foo"):
        gen.collect_third_party_dev_deps(packages)


def test_collect_third_party_dev_deps_with_real_config() -> None:
    """Sanity check against the real repo: the known third-party dev deps come back, deduped."""
    _shared, packages_config = gen.load_configs()
    packages = packages_config["packages"]

    result = gen.collect_third_party_dev_deps(packages)

    assert ("pytest>=9.0.3", "mloda-testing") in result
    assert ("opentelemetry-sdk>=1.30,<2", "mloda-community-otel") in result
    assert not any(req == "mloda-testing" for req, _owner in result), (
        "the internal mloda-testing package must never appear as a synced third-party entry"
    )
    assert sum(1 for req, _owner in result if req.startswith("pytest")) == 1, (
        f"pytest>=9.0.3 is declared by both mloda-testing and mloda-community-otel and must dedupe: {result}"
    )


# --- update_root_dev_dependencies -------------------------------------------------------------


def test_update_root_dev_dependencies_check_detects_missing_third_party_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check mode must report failure, without writing, when a third-party dev dep is missing."""
    real_root_before = _ROOT_PYPROJECT.read_text()
    _shared, packages_config = gen.load_configs()
    packages = packages_config["packages"]

    sandbox_root = _sandbox_root_pyproject(["tox", "pytest", "ruff"])
    (tmp_path / "pyproject.toml").write_text(sandbox_root)
    monkeypatch.chdir(tmp_path)

    ok, message = gen.update_root_dev_dependencies(packages, check=True)

    assert ok is False, f"expected check mode to detect the missing opentelemetry-sdk entry, got: {message!r}"
    assert (tmp_path / "pyproject.toml").read_text() == sandbox_root, "check mode must never write"
    assert _ROOT_PYPROJECT.read_text() == real_root_before


def test_update_root_dev_dependencies_write_mode_adds_missing_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write mode adds the missing third-party entry, leaves hand-authored lines untouched, and is idempotent."""
    real_root_before = _ROOT_PYPROJECT.read_text()
    _shared, packages_config = gen.load_configs()
    packages = packages_config["packages"]

    hand_authored = ["tox", "pytest", "ruff"]
    (tmp_path / "pyproject.toml").write_text(_sandbox_root_pyproject(hand_authored))
    monkeypatch.chdir(tmp_path)

    ok, message = gen.update_root_dev_dependencies(packages, check=False)
    assert ok is True and message == "updated"

    updated_text = (tmp_path / "pyproject.toml").read_text()
    for entry in hand_authored:
        assert f'"{entry}",' in updated_text, f"hand-authored entry {entry!r} must survive untouched"

    dev_list = _dev_list(updated_text)
    assert dev_list[: len(hand_authored)] == hand_authored, "hand-authored lines must keep their original order"
    assert "opentelemetry-sdk>=1.30,<2" in dev_list
    assert len(dev_list) == len(set(dev_list)), f"no duplicate entries expected: {dev_list}"

    ok2, message2 = gen.update_root_dev_dependencies(packages, check=False)
    assert (ok2, message2) == (True, "up-to-date"), "a second write-mode run must be a no-op"
    assert (tmp_path / "pyproject.toml").read_text() == updated_text

    ok3, message3 = gen.update_root_dev_dependencies(packages, check=True)
    assert (ok3, message3) == (True, "up-to-date"), "check mode on an already-synced file must report up-to-date"

    assert _ROOT_PYPROJECT.read_text() == real_root_before


def test_update_root_dev_dependencies_write_mode_does_not_duplicate_hand_authored_pytest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A package's own pytest>=9.0.3 dev requirement must not duplicate an existing hand-authored pin."""
    real_root_before = _ROOT_PYPROJECT.read_text()
    _shared, packages_config = gen.load_configs()
    packages = packages_config["packages"]

    hand_authored = ["tox", "pytest>=9.0.3", "ruff"]
    (tmp_path / "pyproject.toml").write_text(_sandbox_root_pyproject(hand_authored))
    monkeypatch.chdir(tmp_path)

    ok, message = gen.update_root_dev_dependencies(packages, check=False)
    assert ok is True

    dev_list = _dev_list((tmp_path / "pyproject.toml").read_text())
    assert dev_list.count("pytest>=9.0.3") == 1, f"pytest>=9.0.3 must appear exactly once: {dev_list}"
    assert "opentelemetry-sdk>=1.30,<2" in dev_list

    assert _ROOT_PYPROJECT.read_text() == real_root_before


def test_update_root_dev_dependencies_write_mode_noop_when_everything_covered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No dangling marker/empty auto section when every third-party dep is already hand-authored."""
    real_root_before = _ROOT_PYPROJECT.read_text()
    _shared, packages_config = gen.load_configs()
    packages = packages_config["packages"]

    fully_covered = _sandbox_root_pyproject(["tox", "pytest", "opentelemetry-sdk>=1.30,<2"])
    (tmp_path / "pyproject.toml").write_text(fully_covered)
    monkeypatch.chdir(tmp_path)

    ok_check, message_check = gen.update_root_dev_dependencies(packages, check=True)
    assert (ok_check, message_check) == (True, "up-to-date")
    assert (tmp_path / "pyproject.toml").read_text() == fully_covered

    ok_write, message_write = gen.update_root_dev_dependencies(packages, check=False)
    assert (ok_write, message_write) == (True, "up-to-date")
    assert (tmp_path / "pyproject.toml").read_text() == fully_covered, (
        "write mode must not introduce a dangling marker or reorder anything when nothing needs syncing"
    )

    assert _ROOT_PYPROJECT.read_text() == real_root_before


# --- main() wiring -----------------------------------------------------------------------------


def test_main_check_mode_fails_when_root_dev_extras_are_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """gen.main() --check must fail when a package's third-party dev dep is missing from root.

    Neither sandbox has real per-package pyproject.toml files or source trees, so every package
    reports "missing" as baseline noise in both runs. The two sandboxes differ only in whether
    the root dev list already contains opentelemetry-sdk; comparing error counts isolates the
    effect of the dev-deps sync from that shared noise, rather than requiring an implausible
    full-repo mirror.
    """
    real_root_before = _ROOT_PYPROJECT.read_text()

    shared, packages_config = gen.load_configs()
    packages = packages_config["packages"]
    core_dep = shared["defaults"]["core_dependency"]
    workspace_section = gen.generate_workspace_members(packages)

    def _root_pyproject(dev_entries: list[str]) -> str:
        return (
            "[project]\n"
            'name = "sandbox-root"\n'
            "dependencies = [\n"
            f'    "{core_dep}",\n'
            "]\n"
            "\n"
            "[project.optional-dependencies]\n"
            f"{_dev_array_block(dev_entries)}\n"
            "\n"
            f"{workspace_section}\n"
        )

    healthy_dir, stale_dir = tmp_path / "healthy", tmp_path / "stale"
    for sandbox_dir, dev_entries in (
        (healthy_dir, ["tox", "pytest", "opentelemetry-sdk>=1.30,<2"]),
        (stale_dir, ["tox", "pytest"]),
    ):
        _copy_real_configs(sandbox_dir)
        (sandbox_dir / "pyproject.toml").write_text(_root_pyproject(dev_entries))

    monkeypatch.setattr(sys, "argv", ["generate_pyproject.py", "--check"])

    monkeypatch.chdir(healthy_dir)
    gen.main()
    healthy_error_count = capsys.readouterr().out.count("\n  - ")

    monkeypatch.chdir(stale_dir)
    stale_return_code = gen.main()
    stale_out = capsys.readouterr().out

    assert stale_return_code == 1, (
        "gen.main() --check must return 1 when a package's third-party dev dependency (e.g. "
        f"opentelemetry-sdk) is missing from root pyproject.toml's dev extras, got {stale_return_code!r}"
    )
    assert stale_out.count("\n  - ") == healthy_error_count + 1, (
        "the stale dev extras must add exactly one root-level error on top of the baseline "
        f"unrelated per-package noise (baseline count={healthy_error_count}); stale output:\n{stale_out}"
    )

    assert _ROOT_PYPROJECT.read_text() == real_root_before
