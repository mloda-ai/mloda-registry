"""The gate env splits into a pytest part and a lint part that CI runs as parallel jobs.

Locally ``tox -e python3XX`` stays the whole gate: pytest, ``ruff format --check``, ``ruff check``,
``mypy --strict`` and ``bandit``. The ``tests`` factor (``python3XX-tests``) selects only pytest and the
``lint`` factor (``python3XX-lint``) only the four lint steps, through tox 4 factor conditions on the
shared ``[testenv]`` commands. CI's ``build`` job runs both parts for every gate env, so the lint steps
no longer sit on the critical path after pytest.

tox.ini and ci.yaml are read as text (configparser for tox.ini, a small reader for the one matrix block),
and the factor conditions and the GitHub matrix are evaluated here rather than by running tox.
"""

from __future__ import annotations

import configparser
import itertools
import re
import shlex
from collections import Counter
from pathlib import Path

import pytest

from tests.tox_ini import split_factor_condition

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOX_INI = _REPO_ROOT / "tox.ini"
_CI_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yaml"

_PYTEST = "pytest"
_LINT_STEPS = ("ruff format --check", "ruff check", "mypy --strict", "bandit")
_GATE_STEPS = (_PYTEST, *_LINT_STEPS)
_PARTS = ("tests", "lint")

_GATE_ENV = re.compile(r"python3\d+")
_MATRIX_EXPR = re.compile(r"\$\{\{\s*matrix\.(?P<key>[\w-]+)\s*\}\}")


def _tox() -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(_TOX_INI)
    return parser


def _expand_env_name(name: str) -> list[str]:
    """Env names a (possibly generative) section name declares: "py3{11,12}-{a,b}" or "py3{11-12}"."""
    pieces = re.split(r"\{([^{}]*)\}", name)
    choices: list[list[str]] = []
    for index, piece in enumerate(pieces):
        if index % 2 == 0:
            choices.append([piece])
            continue
        alternatives: list[str] = []
        for item in (part.strip() for part in piece.split(",")):
            bounds = re.fullmatch(r"(\d+)-(\d+)", item)
            if bounds:
                alternatives.extend(str(n) for n in range(int(bounds.group(1)), int(bounds.group(2)) + 1))
            else:
                alternatives.append(item)
        choices.append(alternatives)
    return ["".join(combination) for combination in itertools.product(*choices)]


def _declared_envs() -> dict[str, str]:
    """Every env name tox.ini declares, mapped to the section that declares it."""
    declared: dict[str, str] = {}
    for section in _tox().sections():
        if section.startswith("testenv:"):
            for env in _expand_env_name(section.removeprefix("testenv:")):
                declared[env] = section
    return declared


def _gate_envs() -> list[str]:
    envs = sorted(env for env in _declared_envs() if _GATE_ENV.fullmatch(env))
    assert envs, "tox.ini declares no python3XX gate env"
    return envs


def _condition_holds(condition: str | None, factors: set[str]) -> bool:
    """tox 4 semantics: "," separates alternatives, "-" joins factors that must all hold, "!x" means x is absent."""
    if condition is None:
        return True
    return any(
        all(factor[1:] not in factors if factor.startswith("!") else factor in factors for factor in alt.split("-"))
        for alt in condition.split(",")
    )


def _step(command: str) -> str:
    """The gate step a command runs, or the raw command when it is none of them."""
    tokens = shlex.split(command)
    if tokens[:1] == ["pytest"] or (tokens[:2] == ["sh", "-c"] and re.search(r"\bpytest\b", tokens[2])):
        return _PYTEST
    if tokens[:2] == ["ruff", "format"] and "--check" in tokens:
        return "ruff format --check"
    if tokens[:2] == ["ruff", "check"]:
        return "ruff check"
    if tokens[:1] == ["mypy"] and "--strict" in tokens:
        return "mypy --strict"
    if tokens[:1] == ["bandit"]:
        return "bandit"
    return command


def _steps_run_by(env: str) -> list[str]:
    """The gate steps ``tox -e <env>`` runs, in order: its own commands, else the factor-filtered [testenv] ones."""
    parser = _tox()
    section = _declared_envs().get(env)
    own = section is not None and parser.has_option(section, "commands")
    commands = parser[section if own and section else "testenv"]["commands"]
    factors = set(env.split("-"))
    steps = []
    for line in commands.splitlines():
        if not line.strip():
            continue
        condition, command = split_factor_condition(line)
        if _condition_holds(condition, factors):
            steps.append(_step(command))
    return steps


def test_bare_gate_env_runs_all_five_steps() -> None:
    """``tox -e python3XX`` stays the full local gate."""
    for env in _gate_envs():
        assert Counter(_steps_run_by(env)) == Counter(_GATE_STEPS), (
            f"tox -e {env} must run each of {list(_GATE_STEPS)} exactly once, runs {_steps_run_by(env)}"
        )


def test_tests_factor_selects_only_pytest() -> None:
    """``tox -e python3XX-tests`` runs pytest and none of the lint steps."""
    for env in _gate_envs():
        steps = _steps_run_by(f"{env}-tests")
        assert steps == [_PYTEST], (
            f"tox -e {env}-tests must run only pytest, runs {steps}: prefix the four lint lines of "
            "[testenv] commands with the factor condition '!tests: '"
        )


def test_lint_factor_selects_only_the_four_lint_steps() -> None:
    """``tox -e python3XX-lint`` runs the four lint steps and not pytest."""
    for env in _gate_envs():
        steps = _steps_run_by(f"{env}-lint")
        assert Counter(steps) == Counter(_LINT_STEPS), (
            f"tox -e {env}-lint must run each of {list(_LINT_STEPS)} exactly once, runs {steps}: prefix the "
            "pytest line of [testenv] commands with the factor condition '!lint: '"
        )


def test_split_parts_together_run_exactly_the_gate() -> None:
    """No gate step is lost or run twice when CI runs the two parts instead of the bare gate."""
    for env in _gate_envs():
        tests_steps = _steps_run_by(f"{env}-tests")
        lint_steps = _steps_run_by(f"{env}-lint")
        overlap = sorted(set(tests_steps) & set(lint_steps))
        assert not overlap, f"{env}-tests and {env}-lint both run {overlap}"
        assert Counter(tests_steps + lint_steps) == Counter(_steps_run_by(env)), (
            f"{env}-tests ({tests_steps}) plus {env}-lint ({lint_steps}) must run exactly what {env} runs "
            f"({_steps_run_by(env)})"
        )


def test_every_gate_env_declares_tests_and_lint_variants() -> None:
    """Each python3XX env has declared -tests and -lint envs, so ``tox -e python3XX-<part>`` resolves."""
    declared = _declared_envs()
    missing = [f"{env}-{part}" for env in _gate_envs() for part in _PARTS if f"{env}-{part}" not in declared]
    assert not missing, (
        f"tox.ini declares no {missing}: add a generative [testenv:python3{{11,...}}-{{tests,lint}}] section "
        "next to the per-version sections"
    )


def _build_job() -> str:
    """Text of the ci.yaml ``build`` job, up to the next job."""
    match = re.search(r"^  build:\n(.*?)(?=^  [\w-]+:\n|\Z)", _CI_WORKFLOW.read_text(), re.MULTILINE | re.DOTALL)
    assert match is not None, ".github/workflows/ci.yaml has no 'build' job"
    return match.group(1)


def _scalar(value: str) -> str:
    return value.strip().strip("\"'")


def _matrix_combinations() -> list[dict[str, str]]:
    """The build job's matrix combinations, expanded like GitHub does (axes product, then include).

    Reads only the subset the build matrix uses: axes as flow lists ("key: [a, b]") or block lists, and an
    ``include`` list of flat mappings.
    """
    job = _build_job()
    header = re.search(r"^(?P<indent>[ \t]*)matrix:[ \t]*\n", job, re.MULTILINE)
    assert header is not None, "the ci.yaml build job has no strategy matrix"
    lines: list[str] = []
    for line in job[header.end() :].splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if len(line) - len(line.lstrip()) <= len(header.group("indent")):
            break
        lines.append(line)

    key_indent = len(lines[0]) - len(lines[0].lstrip())
    axes: dict[str, list[str]] = {}
    include: list[dict[str, str]] = []
    current = ""
    for line in lines:
        stripped = line.strip()
        if len(line) - len(line.lstrip()) == key_indent:
            current, _, value = stripped.partition(":")
            if current == "exclude":
                pytest.fail("the ci.yaml build matrix uses 'exclude', which this test's matrix reader does not model")
            if current != "include":
                axes[current] = [_scalar(item) for item in value.strip().strip("[]").split(",") if item.strip()]
        elif current == "include":
            if stripped.startswith("- "):
                include.append({})
                stripped = stripped[2:]
            key, _, value = stripped.partition(":")
            include[-1][key.strip()] = _scalar(value)
        else:
            axes[current].append(_scalar(stripped.removeprefix("- ")))

    if not axes:
        return [dict(entry) for entry in include]
    combinations = [dict(zip(axes, values)) for values in itertools.product(*axes.values())]
    originals = [dict(combination) for combination in combinations]
    for entry in include:
        # An include entry extends every original combination it does not overwrite, else it adds its own.
        targets = [
            combination
            for combination, original in zip(combinations, originals)
            if all(original.get(key, value) == value for key, value in entry.items())
        ]
        for combination in targets:
            combination.update(entry)
        if not targets:
            combinations.append(dict(entry))
    return combinations


def _tox_run_template() -> str:
    runs = re.findall(r"^\s*run:\s*tox -e (.+?)\s*$", _build_job(), re.MULTILINE)
    assert len(runs) == 1, f"expected one 'run: tox -e ...' step in the ci.yaml build job, found {runs}"
    return str(runs[0])


def _render(template: str, combination: dict[str, str]) -> str:
    missing = sorted({key for key in _MATRIX_EXPR.findall(template) if key not in combination})
    assert not missing, f"the build matrix combination {combination} has no {missing}"
    return _MATRIX_EXPR.sub(lambda match: combination[match.group("key")], template)


def test_ci_build_run_step_appends_the_part_factor() -> None:
    """The build job runs one part per job: ``tox -e <toxenv>-<part>``."""
    template = _tox_run_template()
    assert re.fullmatch(r"\$\{\{\s*matrix\.toxenv\s*\}\}-\$\{\{\s*matrix\.part\s*\}\}", template), (
        f"the ci.yaml build job runs 'tox -e {template}', expected 'tox -e ${{{{ matrix.toxenv }}}}-${{{{ matrix.part }}}}'"
    )


def test_ci_build_matrix_runs_both_parts_of_every_gate_env() -> None:
    """Every python3XX env in tox.ini runs as a tests job and a lint job, each exactly once."""
    template = _tox_run_template()
    combinations = _matrix_combinations()
    parts = [combination.get("part") for combination in combinations]
    assert all(part in _PARTS for part in parts), (
        f"every build matrix combination needs a 'part' in {list(_PARTS)}, got {parts}: add 'part: [tests, lint]'"
    )
    ran = sorted(_render(template, combination) for combination in combinations)
    expected = sorted(f"{env}-{part}" for env in _gate_envs() for part in _PARTS)
    assert ran == expected, f"the ci.yaml build job runs tox envs {ran}, expected {expected}"


def test_ci_build_matrix_python_version_matches_toxenv() -> None:
    """Each build job sets up the interpreter its toxenv names (python312 on 3.12)."""
    for combination in _matrix_combinations():
        toxenv = combination.get("toxenv", "")
        assert _GATE_ENV.fullmatch(toxenv), f"build matrix combination {combination} has no python3XX toxenv"
        expected = f"3.{toxenv.removeprefix('python3')}"
        assert combination.get("python-version") == expected, (
            f"build matrix combination {combination} must set python-version {expected!r} for {toxenv}"
        )


def test_ci_build_keeps_pytest_workers() -> None:
    """The tests part keeps its pytest-xdist worker count in CI."""
    assert re.search(r"^\s*PYTEST_WORKERS:\s*\S", _build_job(), re.MULTILINE), (
        "the ci.yaml build job no longer sets PYTEST_WORKERS for tox"
    )
