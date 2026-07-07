"""Lint for HashableDict reappearing near credential / BaseInputData data-access construction.

Regression guard for GitHub issue #270: mloda 0.9.0 rejects HashableDict in
credential / BaseInputData data-access construction. This flags any HashableDict
token that appears on or within a few lines of a credential marker, while leaving
legitimate HashableDict usage (Options group values, ApiInputDataCollection,
provider export/import) untouched.

Run: python scripts/lint_credentials.py
Exit code: 1 if any issues found, 0 otherwise.
"""

import re
import sys
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
SCAN_ROOT: Path = REPO_ROOT

SCAN_SUFFIXES: tuple[str, ...] = (".py", ".md")

EXCLUDE_DIRS: frozenset[str] = frozenset(
    {".git", ".tox", ".venv", "venv", "__pycache__", "node_modules", "build", "dist"}
)

SELF_FILES: frozenset[str] = frozenset({"scripts/lint_credentials.py", "tests/test_end2end/test_lint_credentials.py"})

HASHABLE_DICT_RE = re.compile(r"\bHashableDict\b")

CREDENTIAL_MARKER_RE = re.compile(
    r"\bcredentials\b|\badd_credentials\b|\bDataAccessCollection\b|[\"']BaseInputData[\"']"
)


def scan_content(rel_path: str, content: str, window: int = 2) -> list[str]:
    """Flag each HashableDict line that has a credential marker within +/- window lines.

    Pure: takes text, returns violation strings. Each violation contains the
    substrings ``HashableDict`` and ``credential`` plus a ``{rel_path}:{lineno}:``
    locator, where lineno is the 1-based line of the HashableDict token.
    """
    lines = content.splitlines()
    errors: list[str] = []
    for index, line in enumerate(lines):
        if not HASHABLE_DICT_RE.search(line):
            continue
        lo = max(0, index - window)
        hi = min(len(lines), index + window + 1)
        if any(CREDENTIAL_MARKER_RE.search(lines[near]) for near in range(lo, hi)):
            lineno = index + 1
            errors.append(f"{rel_path}:{lineno}: HashableDict used in credential / data-access construction")
    return errors


def _is_excluded_dir(part: str) -> bool:
    return part in EXCLUDE_DIRS or part.endswith(".egg-info")


def find_credential_hashable_dict(root: Path) -> list[str]:
    """Walk root for .py/.md files and return sorted credential-context HashableDict violations."""
    errors: list[str] = []
    for path in root.rglob("*"):
        if path.suffix not in SCAN_SUFFIXES or not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(_is_excluded_dir(part) for part in rel.parts[:-1]):
            continue
        rel_posix = rel.as_posix()
        if rel_posix in SELF_FILES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        errors.extend(scan_content(rel_posix, content))
    return sorted(errors)


def main() -> int:
    errors = find_credential_hashable_dict(SCAN_ROOT)
    if errors:
        print(f"Found {len(errors)} credential-context HashableDict issue(s):\n")
        for error in errors:
            print(f"  {error}")
        return 1

    print("No credential-context HashableDict usage found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
