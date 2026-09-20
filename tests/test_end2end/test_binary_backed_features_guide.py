"""Doc-drift guard for ``docs/guides/feature-group-patterns/28-binary-backed-features.md``, for the tox.ini and
ci.yaml wiring of the real-wheel suite, and for the documented ``uv sync`` setup commands. No mktestdocs/sybil-style
execution of guide code fences is wired in this repo (confirmed), so these are lightweight grep-based checks
instead of executed doctests."""

from __future__ import annotations

import configparser
import re
import shlex
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GUIDE_PATH = _REPO_ROOT / "docs" / "guides" / "feature-group-patterns" / "28-binary-backed-features.md"
_TOX_INI = _REPO_ROOT / "tox.ini"
_CI_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yaml"
_README = _REPO_ROOT / "README.md"
_CONTRIBUTING = _REPO_ROOT / "CONTRIBUTING.md"
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"
_PACKAGING_DOC = _REPO_ROOT / "docs" / "packaging.md"
# AGENTS.md is a byte-identical copy of CLAUDE.md, already covered by tests/test_end2end/test_agent_guidance.py.
_SETUP_COMMAND_DOCS = (_README, _CONTRIBUTING, _CLAUDE_MD, _PACKAGING_DOC)


def test_complete_example_sets_binary_wheel_distribution() -> None:
    """The real ``BinaryExampleFeatureGroup`` sets ``BINARY_WHEEL_DISTRIBUTION`` (see
    ``mloda/enterprise/feature_groups/binary_example/binary_example_feature_group.py``); the
    guide's "Complete Example" code block must show it too, alongside ``BINARY_PLUGIN_ID``, or a
    reader copying the example ships a package with no version-pinned optional wheel dependency."""
    content = _GUIDE_PATH.read_text(encoding="utf-8")
    example_start = content.index("## Complete Example")
    remainder = content[example_start + len("## Complete Example") :]
    next_heading = remainder.find("\n## ")
    example = remainder if next_heading == -1 else remainder[:next_heading]
    assert "BINARY_PLUGIN_ID" in example, "fixture assumption: the Complete Example sets BINARY_PLUGIN_ID"
    assert "BINARY_WHEEL_DISTRIBUTION" in example, (
        "the Complete Example code block must also set BINARY_WHEEL_DISTRIBUTION, not just mention "
        "it elsewhere in the guide (e.g. the Key Characteristic table or Packaging Rules prose)"
    )


def _section(content: str, heading: str, boundary_markers: tuple[str, ...]) -> str:
    """The text of the section under ``heading``, up to the closest of ``boundary_markers``."""
    start = content.index(heading)
    remainder = content[start + len(heading) :]
    boundaries = [pos for pos in (remainder.find(marker) for marker in boundary_markers) if pos != -1]
    end = min(boundaries) if boundaries else -1
    return remainder if end == -1 else remainder[:end]


def test_packaging_rules_section_states_how_to_install_the_wheel() -> None:
    """The Packaging Rules section gives the plain install command for the wheel."""
    content = _GUIDE_PATH.read_text(encoding="utf-8")
    section = _section(content, "## Packaging Rules", ("\n## ",))
    assert "pip install mloda-example-binary" in section, (
        "the Packaging Rules section must contain `pip install mloda-example-binary`; section was:\n" + section
    )


def test_against_the_real_wheel_section_shows_the_standalone_suite_command() -> None:
    """The "### Against the real wheel" section shows the command that runs the real-wheel suite on its own."""
    content = _GUIDE_PATH.read_text(encoding="utf-8")
    section = _section(content, "### Against the real wheel", ("\n## ", "\n### "))
    assert "pytest tests/test_binary_model_real" in section, (
        'the "Against the real wheel" section must contain `pytest tests/test_binary_model_real`; section was:\n'
        + section
    )


def test_against_the_real_wheel_section_shows_the_uv_install_command() -> None:
    """The "### Against the real wheel" section gives the uv form of the install, as the dev venv has no pip."""
    content = _GUIDE_PATH.read_text(encoding="utf-8")
    section = _section(content, "### Against the real wheel", ("\n## ", "\n### "))
    assert "uv pip install mloda-example-binary" in section, (
        'the "Against the real wheel" section must contain `uv pip install mloda-example-binary`; section was:\n'
        + section
    )


def test_real_wheel_tox_env_installs_the_wheel_and_probes_the_import() -> None:
    """The real-wheel env installs the wheel extra, probes the import first and runs the whole suite directory."""
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(_TOX_INI)
    assert parser.has_section("testenv:real-wheel"), (
        "tox.ini must define [testenv:real-wheel] to run tests/test_binary_model_real against the real wheel"
    )
    env = parser["testenv:real-wheel"]
    extras = re.split(r"[\s,]+", env.get("extras", ""))
    assert "wheel" in extras, f"[testenv:real-wheel] must install the `wheel` extra; extras were: {extras}"
    commands = env.get("commands", "").splitlines()
    probe = next((i for i, line in enumerate(commands) if "import example_binary" in line), None)
    run = next((i for i, line in enumerate(commands) if "pytest" in line), None)
    assert probe is not None, f"[testenv:real-wheel] must probe `import example_binary`; commands were: {commands}"
    assert run is not None and probe < run, (
        f"the `import example_binary` probe must run before the pytest line; commands were: {commands}"
    )
    tokens = shlex.split(commands[run])
    assert "tests/test_binary_model_real/" in tokens or "tests/test_binary_model_real" in tokens, (
        f"the pytest line must run the whole suite directory tests/test_binary_model_real/, not a single file; "
        f"pytest line was: {commands[run]}"
    )


def test_ci_runs_the_real_wheel_tox_env() -> None:
    """ci.yaml has a non-comment `run: tox -e real-wheel` line, so the job cannot be commented out unnoticed."""
    workflow = _CI_WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r"^\s*(?:-\s+)?run:\s*tox -e real-wheel\s*$", workflow, re.MULTILINE), (
        "ci.yaml must have a non-comment step that runs `tox -e real-wheel`"
    )


def test_against_the_real_wheel_section_shows_the_tox_env_command() -> None:
    """The "### Against the real wheel" section shows the tox env that runs the real-wheel suite."""
    content = _GUIDE_PATH.read_text(encoding="utf-8")
    section = _section(content, "### Against the real wheel", ("\n## ", "\n### "))
    assert "tox -e real-wheel" in section, (
        'the "Against the real wheel" section must contain `tox -e real-wheel`; section was:\n' + section
    )


def test_documented_uv_sync_commands_use_the_gate_flags_not_all_extras() -> None:
    """Every documented ``uv sync`` setup command uses the gate's flags (``--all-packages --extra dev``). With
    ``--all-extras`` uv also installs the ``wheel`` extra of ``mloda-enterprise-binary-example``, i.e. the real
    ``mloda-example-binary`` wheel, and the suites assume that wheel is absent (two tripwire tests fail if it is
    installed), so a fresh checkout that follows the docs would get red tests."""
    for path in _SETUP_COMMAND_DOCS:
        name = path.relative_to(_REPO_ROOT).as_posix()
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        commands = [line for line in lines if line.startswith("uv sync")]
        assert commands, (
            f"{name} must document the dev setup as a line starting with `uv sync` (a reworded or removed "
            "command would otherwise pass this guard vacuously)"
        )
        for command in commands:
            assert "--all-extras" not in command, (
                f"{name}: `{command}` uses `--all-extras`, which installs the real binary wheel (the `wheel` "
                "extra) into the venv, but the suites assume it is absent; use the gate's own flags "
                "`--all-packages --extra dev`"
            )
            assert "--extra dev" in command, (
                f"{name}: `{command}` must contain `--extra dev`; `--all-extras` would install the real binary "
                "wheel (the `wheel` extra) into the venv, but the suites assume it is absent; use the gate's "
                "own flags `--all-packages --extra dev`"
            )
