"""Tests that the weekly verification workflow checks out the release it verifies, not main.

The published set (``config/packages.toml`` read through ``scripts/published_packages.py``), the verify
scripts and ``tox.ini`` all come from the checkout. If the workflow checks out main and then verifies the
latest release, a package newly flagged ``published = true`` on main fails the job before its first
release. The workflow must therefore resolve the latest release first and then check out that release's
tag (tags are the bare version, ``tagFormat: ${version}`` in ``.releaserc.yaml``), so the published set
and verify scripts describe the release under test. Since ``gh release view`` then runs with no git
repository on disk, it must name the repository explicitly.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW_PATH = _REPO_ROOT / ".github" / "workflows" / "verify-published.yaml"

_CHECKOUT_STEP = "Check out the release under test"
_GET_VERSION_STEP = "Get latest release version"
_GET_VERSION_ID = "get_version"

# The job's 'steps:' key; every step is a list item one level below it.
_STEPS_KEY_RE = re.compile(r"^[ \t]*steps:[ \t]*\n", re.MULTILINE)
# A step list item (named or not), up to the next item at the same indent or end of file.
_STEP_ITEM_RE = re.compile(
    r"^(?P<indent>[ \t]*)- (?P<text>.*?)(?=^(?P=indent)- |\Z)",
    re.MULTILINE | re.DOTALL,
)
_NAME_RE = re.compile(r"^(?:- )?\s*name:\s*(?P<name>[^\n]+)$", re.MULTILINE)
_ID_RE = re.compile(r"^\s*id:\s*(?P<id>[^\n]+)$", re.MULTILINE)
_CHECKOUT_USES_RE = re.compile(r"^(?:- )?\s*uses:\s*actions/checkout@", re.MULTILINE)
_REF_RE = re.compile(r"^\s*ref:\s*(?P<ref>[^\n]+)$", re.MULTILINE)
_GH_REPO_ENV_RE = re.compile(r"^\s*GH_REPO:\s*\$\{\{\s*github\.repository\s*\}\}\s*$", re.MULTILINE)
_REPO_FLAG_RE = re.compile(r"--repo[ =]\"?\$\{\{\s*github\.repository\s*\}\}")


def _steps() -> list[str]:
    """Text of every step in the workflow's job, in order, named or not."""
    text = _WORKFLOW_PATH.read_text()
    steps_key = _STEPS_KEY_RE.search(text)
    assert steps_key is not None, ".github/workflows/verify-published.yaml has no 'steps:' key"
    return [match.group("text") for match in _STEP_ITEM_RE.finditer(text, steps_key.end())]


def _step_name(step: str) -> str | None:
    match = _NAME_RE.search(step)
    return match.group("name").strip() if match else None


def _step_index(name: str) -> int:
    names = [_step_name(step) for step in _steps()]
    assert name in names, f".github/workflows/verify-published.yaml has no '- name: {name}' step (steps: {names})"
    return names.index(name)


def _checkout_indices() -> list[int]:
    return [index for index, step in enumerate(_steps()) if _CHECKOUT_USES_RE.search(step)]


def test_checkout_step_is_named_and_uses_checkout_action() -> None:
    """The checkout is a named step so its position relative to 'Get latest release version' is explicit."""
    step = _steps()[_step_index(_CHECKOUT_STEP)]
    assert _CHECKOUT_USES_RE.search(step), (
        f".github/workflows/verify-published.yaml step '{_CHECKOUT_STEP}' does not use 'actions/checkout@v4': {step!r}"
    )
    assert "actions/checkout@v4" in step, (
        f".github/workflows/verify-published.yaml step '{_CHECKOUT_STEP}' must pin 'actions/checkout@v4': {step!r}"
    )


def test_checkout_step_refs_the_resolved_release() -> None:
    """Without a 'ref:' the checkout is main, so the published set and verify scripts describe main."""
    step = _steps()[_step_index(_CHECKOUT_STEP)]
    ref_match = _REF_RE.search(step)
    assert ref_match is not None, (
        f".github/workflows/verify-published.yaml step '{_CHECKOUT_STEP}' has no 'ref:', so it checks out "
        "main instead of the release tag whose packages the job verifies"
    )
    assert f"steps.{_GET_VERSION_ID}.outputs." in ref_match.group("ref"), (
        f".github/workflows/verify-published.yaml step '{_CHECKOUT_STEP}' has 'ref: {ref_match.group('ref')}', "
        f"which is not derived from the '{_GET_VERSION_STEP}' step's output "
        f"('steps.{_GET_VERSION_ID}.outputs.<name>')"
    )


def test_release_checkout_is_the_only_checkout_and_runs_after_get_version() -> None:
    """The release must be resolved before the checkout, and no un-ref'd checkout may leave main on disk."""
    steps = _steps()
    get_version_index = _step_index(_GET_VERSION_STEP)
    id_match = _ID_RE.search(steps[get_version_index])
    assert id_match is not None and id_match.group("id").strip() == _GET_VERSION_ID, (
        f".github/workflows/verify-published.yaml step '{_GET_VERSION_STEP}' must keep 'id: {_GET_VERSION_ID}'"
    )

    checkouts = _checkout_indices()
    early_checkouts = [index for index in checkouts if index < get_version_index]
    assert early_checkouts == [], (
        f".github/workflows/verify-published.yaml runs actions/checkout at step position(s) {early_checkouts}, "
        f"before '{_GET_VERSION_STEP}' (position {get_version_index}); that checkout cannot target the release "
        "tag and leaves main on disk for the published set and verify scripts"
    )
    checkout_index = _step_index(_CHECKOUT_STEP)
    assert checkouts == [checkout_index], (
        f".github/workflows/verify-published.yaml has actions/checkout at step position(s) {checkouts}, "
        f"expected exactly one, the '{_CHECKOUT_STEP}' step at position {checkout_index}; any other checkout "
        "leaves main (or a second tree) on disk for the published set and verify scripts"
    )

    tox_steps = [index for index, step in enumerate(steps) if re.search(r"\btox -e\b", step)]
    assert tox_steps, "fixture assumption: the workflow has steps that run 'tox -e ...'"
    early = [_step_name(steps[index]) for index in tox_steps if index < checkout_index]
    assert early == [], (
        f".github/workflows/verify-published.yaml steps {early} run tox before '{_CHECKOUT_STEP}', so they "
        "read a tox.ini and verify scripts that are not the release under test"
    )


def test_get_version_names_the_repository_explicitly() -> None:
    """'gh release view' runs before any checkout, so it cannot infer the repository from a local git repo."""
    step = _steps()[_step_index(_GET_VERSION_STEP)]
    assert _GH_REPO_ENV_RE.search(step) or _REPO_FLAG_RE.search(step), (
        f".github/workflows/verify-published.yaml step '{_GET_VERSION_STEP}' neither sets "
        "'GH_REPO: ${{ github.repository }}' in its env nor passes '--repo ${{ github.repository }}' to "
        "'gh release view'; with no checkout before it, gh has no git repository to infer the repository from"
    )
