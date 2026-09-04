"""Regression tests for two race conditions the sibling ``mloda`` core repo already hit and fixed in its
own release pipeline: a stale ``uv.lock`` after every release, and the ``publish`` job building from
whatever is on ``main`` when it happens to run instead of the exact commit semantic-release tagged.

Both files are read as raw text and checked with regex / substring assertions, matching the convention
in ``test_published_set_single_source.py`` (``_tox_block`` / ``_workflow_build_step``). Neither file is
parsed as structured YAML: GitHub Actions' ``on:`` key is a well-known PyYAML 1.1 boolean-coercion trap.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RELEASE_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "release.yaml"
_RELEASERC = _REPO_ROOT / ".releaserc.yaml"

# An exact-pinned uv install, e.g. "uv==0.11.8".
_PINNED_UV_RE = re.compile(r"\buv==\d+\.\d+\.\d+\b")

# A bare/unpinned "install uv" with no version constraint at all (a floating "uv>=..." range doesn't match).
_BARE_UV_INSTALL_RE = re.compile(r"install\s+uv(?!==)(?!>=)\b")

# 'git rev-parse' and 'refs/tags/' on the SAME non-comment line: a decoy comment mentioning both terms
# elsewhere in the job (while the real command resolves something else, e.g. the branch tip) can't match.
_SHA_CAPTURE_RE = re.compile(r"^[^\n#]*\bgit rev-parse\b[^\n#]*refs/tags/", re.MULTILINE)

# 'uv lock' chained directly onto the generate_pyproject.py invocation with '&&', not just present
# somewhere downstream (a ';'-separated or dead/commented fragment would satisfy a bare substring check).
_LOCK_AFTER_GENERATE_RE = re.compile(r"generate_pyproject\.py\s*&&\s*uv lock")

# The 'if:' guard on a step, matched on its own line so it can't be satisfied by an unrelated 'if:' elsewhere.
_NEW_RELEASE_PUBLISHED_GUARD_RE = re.compile(
    r"^\s*if:\s*steps\.semantic\.outputs\.new_release_published == 'true'\s*$", re.MULTILINE
)


def _prepare_cmd() -> str:
    """The @semantic-release/exec plugin's prepareCmd shell string in .releaserc.yaml."""
    match = re.search(r'"prepareCmd":\s*"(?P<cmd>.*?)",?\n', _RELEASERC.read_text(), re.DOTALL)
    assert match is not None, '.releaserc.yaml has no "prepareCmd" entry'
    return match.group("cmd")


def _git_assets() -> str:
    """Body of the @semantic-release/git plugin's 'assets' array in .releaserc.yaml."""
    match = re.search(r'"@semantic-release/git".*?"assets":\s*\[(?P<body>.*?)\]', _RELEASERC.read_text(), re.DOTALL)
    assert match is not None, ".releaserc.yaml has no '@semantic-release/git' plugin with an 'assets' array"
    return match.group("body")


def _job_block(job_name: str) -> str:
    """Body of a top-level job in the release workflow, from its header to the next job or EOF."""
    match = re.search(
        rf"^  {re.escape(job_name)}:\n(?P<body>.*?)(?=^  [A-Za-z_][\w-]*:[ \t]*(?:#.*)?\n|\Z)",
        _RELEASE_WORKFLOW.read_text(),
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f".github/workflows/release.yaml has no top-level job '{job_name}:'"
    return match.group("body")


def _step_containing(job_body: str, needle: re.Pattern[str]) -> str:
    """Body of the single step within a job body containing a line matched by `needle`.

    Scoping to the individual step (rather than searching the whole job) keeps assertions about that
    step's own attributes, e.g. an 'if:' guard, from being satisfied by an unrelated line elsewhere.
    """
    steps = re.split(r"(?=^      - )", job_body, flags=re.MULTILINE)
    matches = [step for step in steps if needle.search(step)]
    assert len(matches) == 1, (
        f"expected exactly one step matching {needle.pattern!r}, found {len(matches)} in job body: {job_body!r}"
    )
    return matches[0]


def _job_outputs(job_body: str) -> str:
    """Body of a job's 'outputs:' block, scoped to that job's own text."""
    match = re.search(r"^    outputs:\n(?P<body>(?:      .+\n)+)", job_body, re.MULTILINE)
    assert match is not None, "release.yaml job has no 'outputs:' block"
    return match.group("body")


def _job_if(job_body: str) -> str:
    """Body of a job's 'if:' condition, including any block-scalar continuation lines."""
    match = re.search(r"^    if:.*(?:\n      .+)*", job_body, re.MULTILINE)
    assert match is not None, "release.yaml job has no 'if:' condition"
    return match.group(0)


# Bug 1: prepareCmd bumps config/shared.toml, tox.ini and every pyproject.toml but never re-locks, so
# uv.lock's embedded per-package versions silently lag one version behind after every release.


def test_prepare_cmd_locks_after_regenerating_pyproject() -> None:
    """uv.lock embeds every package's version; it must be regenerated after the bump it needs to capture."""
    cmd = _prepare_cmd()
    generate_at = cmd.find("python scripts/generate_pyproject.py")
    assert generate_at != -1, f"prepareCmd never runs scripts/generate_pyproject.py: {cmd!r}"
    lock_at = cmd.find("uv lock")
    assert lock_at != -1, f"prepareCmd never runs 'uv lock', so uv.lock goes stale after every release: {cmd!r}"
    assert lock_at > generate_at, (
        f"prepareCmd runs 'uv lock' before scripts/generate_pyproject.py bumps every pyproject.toml, so "
        f"the lock would capture the pre-release versions: {cmd!r}"
    )
    assert _LOCK_AFTER_GENERATE_RE.search(cmd) is not None, (
        f"prepareCmd does not chain 'uv lock' onto 'generate_pyproject.py' with '&&'; a disconnected "
        f"fragment separated by ';' (or a dead, commented-out 'uv lock') would satisfy the two substring "
        f"checks above without the lock actually running as part of the same command: {cmd!r}"
    )


def test_git_plugin_commits_the_lockfile() -> None:
    """Even a regenerated uv.lock never reaches the release commit unless @semantic-release/git carries it."""
    assets = _git_assets()
    assert '"uv.lock"' in assets, f'.releaserc.yaml @semantic-release/git "assets" does not list "uv.lock": {assets!r}'


# Bug 2: the publish job builds from live main instead of the released commit, with no concurrency guard
# and no pinned uv in the job that will now write uv.lock into an unreviewed, automatically-pushed commit.


def test_github_release_job_pins_uv_to_an_exact_version() -> None:
    """This job's uv will write uv.lock into an unreviewed [skip ci] commit, so it must not float."""
    job = _job_block("github_release")
    assert _PINNED_UV_RE.search(job) is not None, (
        "job 'github_release' installs no exact-pinned uv (e.g. 'uv==0.11.8'); an unpinned uv could "
        "silently change uv.lock's format underneath the [skip ci] release commit"
    )
    assert _BARE_UV_INSTALL_RE.search(job) is None, (
        "job 'github_release' installs uv with no version constraint at all; pin it to an exact version instead"
    )


def test_workflow_declares_a_release_concurrency_group() -> None:
    """Two workflow_dispatch runs racing to publish the same version need a shared concurrency group."""
    text = _RELEASE_WORKFLOW.read_text()
    concurrency_match = re.search(r"^concurrency:\n(?P<body>(?:  .+\n)+)", text, re.MULTILINE)
    assert concurrency_match is not None, (
        ".github/workflows/release.yaml has no top-level 'concurrency:' block guarding overlapping runs"
    )
    body = concurrency_match.group("body")
    assert "group:" in body, f"'concurrency:' block has no 'group:' key: {body!r}"
    assert "cancel-in-progress: false" in body, (
        f"'concurrency:' block must set 'cancel-in-progress: false' so a second run queues instead of "
        f"cancelling one that might already be publishing: {body!r}"
    )
    jobs_match = re.search(r"^jobs:\n", text, re.MULTILINE)
    assert jobs_match is not None, ".github/workflows/release.yaml has no 'jobs:' key"
    assert concurrency_match.start() < jobs_match.start(), "'concurrency:' block must come before 'jobs:'"


def test_github_release_job_declares_new_release_sha_output() -> None:
    """The publish job needs the released SHA passed through as a job output to check it out later."""
    outputs = _job_outputs(_job_block("github_release"))
    assert "new_release_sha: ${{ steps.release_sha.outputs.sha }}" in outputs, (
        f"job 'github_release' outputs does not wire 'new_release_sha' to 'steps.release_sha.outputs.sha'; "
        f"wiring it to any other step's output would silently break the publish job's checkout: {outputs!r}"
    )


def test_github_release_job_captures_the_released_sha_from_the_tag() -> None:
    """The SHA output must come from resolving semantic-release's own tag, not from the branch tip."""
    job = _job_block("github_release")
    assert _SHA_CAPTURE_RE.search(job) is not None, (
        "job 'github_release' has no single line running 'git rev-parse' against 'refs/tags/...'; a decoy "
        "comment mentioning both terms separately does not prove the actual command resolves the release tag"
    )


def test_github_release_job_capture_sha_step_is_guarded_by_new_release_published() -> None:
    """A no-release run must not attempt 'git rev-parse' against an unset $VERSION; guard the step itself."""
    step = _step_containing(_job_block("github_release"), _SHA_CAPTURE_RE)
    assert _NEW_RELEASE_PUBLISHED_GUARD_RE.search(step) is not None, (
        f"step capturing the released SHA has no \"if: steps.semantic.outputs.new_release_published == 'true'\" "
        f"guard of its own, so a no-release run would run 'git rev-parse' against an empty/garbage ref: {step!r}"
    )


def test_publish_job_checks_out_the_released_sha() -> None:
    """Checking out 'main' lets a PR merged mid-release ship silently under the wrong version tag."""
    job = _job_block("publish")
    assert "ref: ${{ needs.github_release.outputs.new_release_sha }}" in job, (
        "job 'publish' checkout step does not pin 'ref:' to '${{ needs.github_release.outputs.new_release_sha }}'"
    )
    assert "ref: main" not in job, "job 'publish' still checks out 'ref: main' instead of the released commit"


def test_publish_job_no_longer_pulls_main() -> None:
    """A 'git pull origin main' after checkout defeats pinning the ref: it re-fetches whatever main became."""
    job = _job_block("publish")
    assert "git pull" not in job, (
        f"job 'publish' still runs 'git pull'; the pinned checkout ref alone must be the source of truth: {job!r}"
    )


def test_publish_job_if_also_requires_a_released_sha() -> None:
    """A skipped/failed SHA-capture step must not silently fall through to publishing from an unpinned ref."""
    condition = _job_if(_job_block("publish"))
    assert "new_release_published" in condition, (
        f"job 'publish' if: condition dropped 'new_release_published': {condition!r}"
    )
    assert "new_release_sha" in condition, (
        f"job 'publish' if: condition checks 'new_release_published' but not 'new_release_sha', so a "
        f"failed or skipped SHA-capture step could still let publishing proceed from an unpinned ref: {condition!r}"
    )


def test_tag_format_matches_the_sha_capture_assumption() -> None:
    """release.yaml resolves 'refs/tags/<version>' literally; a changed tagFormat would silently break it."""
    assert re.search(r"^tagFormat:\s*\$\{version\}\s*$", _RELEASERC.read_text(), re.MULTILINE) is not None, (
        ".releaserc.yaml tagFormat is no longer '${version}', but release.yaml resolves "
        "'refs/tags/$VERSION' literally against the git tag; keep both in sync"
    )
