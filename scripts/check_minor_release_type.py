#!/usr/bin/env python3
"""Require a `minor:` PR title and commit when the OpenLineage extender seam tables or `core_dependency` change."""

from __future__ import annotations

import ast
import os
import re
import subprocess  # nosec
import sys
import tomllib
from collections.abc import Callable
from typing import NamedTuple

SEAM_SOURCE = "mloda/testing/extenders/openlineage.py"
SHARED_CONFIG = "config/shared.toml"
TABLE_NAMES = ("OPENLINEAGE_EXTENDER_SEAMS", "OPENLINEAGE_EXTENDER_ATTRIBUTE_SEAMS")
# Every type has a custom releaseRule in .releaserc.yaml, so `!` does not escalate: `minor!:` still releases a minor.
_MINOR = re.compile(r"^minor(\([^)]+\))?!?: .+")


def seam_tables(source: str) -> str | None:
    """Return a formatting-independent dump of both seam tables, or None if either is missing."""
    values: dict[str, ast.expr] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            value: ast.expr | None = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
            value = node.value
        else:
            continue
        for target in targets:
            if target.id in TABLE_NAMES and value is not None:
                values[target.id] = value
    if any(name not in values for name in TABLE_NAMES):
        return None
    return "\n".join(ast.dump(values[name]) for name in TABLE_NAMES)


def is_minor(subject: str) -> bool:
    return _MINOR.match(subject) is not None


def core_dependency(source: str) -> str | None:
    """Return `[defaults] core_dependency` with whitespace removed, or None if missing or unusable."""
    try:
        defaults = tomllib.loads(source).get("defaults")
    except tomllib.TOMLDecodeError:
        return None
    if not isinstance(defaults, dict):
        return None
    value = defaults.get("core_dependency")
    return re.sub(r"\s+", "", value) if isinstance(value, str) else None


_MINOR_REQUIRED = (
    "A squash merge releases the PR title and a merge or rebase merge releases the commits, so the "
    "PR title and at least one commit must both be typed `minor:`."
)


class Guard(NamedTuple):
    name: str
    path: str
    extract: Callable[[str], str | None]
    reason: str


SEAM_GUARD = Guard(
    "seam tables",
    SEAM_SOURCE,
    seam_tables,
    "mloda-enterprise[openlineage] pins mloda-community-openlineage~={version}, which accepts patch releases, "
    f"so a seam change must bump the minor. {_MINOR_REQUIRED}",
)
CORE_DEPENDENCY_GUARD = Guard(
    "core_dependency",
    SHARED_CONFIG,
    core_dependency,
    "Published packages cap mloda below the next minor, so moving core_dependency must release a minor. "
    + _MINOR_REQUIRED,
)
GUARDS = (SEAM_GUARD, CORE_DEPENDENCY_GUARD)


def evaluate(
    guard: Guard, base_source: str | None, head_source: str | None, pr_title: str, subjects: list[str]
) -> str | None:
    """Return an error message, or None when the change is allowed."""
    head = guard.extract(head_source) if head_source is not None else None
    if head is None:
        return (
            f"{guard.name} not found in {guard.path} at the PR head. "
            "Update scripts/check_minor_release_type.py instead of disabling the check."
        )
    base = guard.extract(base_source) if base_source is not None else None
    if base == head:
        return None
    if is_minor(pr_title) and any(is_minor(s) for s in subjects):
        return None
    return f"The {guard.name} in {guard.path} changed. {guard.reason}"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], capture_output=True, text=True)  # nosec


def main() -> int:
    base_sha, head_sha, pr_title = os.environ["BASE_SHA"], os.environ["HEAD_SHA"], os.environ["PR_TITLE"]
    merge_base = _git("merge-base", base_sha, head_sha)
    if merge_base.returncode != 0:
        print(f"::error ::git merge-base failed for {base_sha} and {head_sha}")
        return 1
    range_base = merge_base.stdout.strip()
    log = _git("log", "--no-merges", "--format=%s", f"{range_base}..{head_sha}")
    if log.returncode != 0:
        print(f"::error ::git log failed for range {range_base}..{head_sha}")
        return 1
    subjects = [s for s in log.stdout.splitlines() if s]
    failed = False
    for guard in GUARDS:
        sources: list[str | None] = []
        for sha in (range_base, head_sha):
            shown = _git("show", f"{sha}:{guard.path}")
            sources.append(shown.stdout if shown.returncode == 0 else None)
        error = evaluate(guard, sources[0], sources[1], pr_title, subjects)
        if error is not None:
            print(f"::error ::{error}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
