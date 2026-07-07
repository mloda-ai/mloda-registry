"""Lint for HashableDict reappearing inside credential / BaseInputData data-access construction.

Regression guard for GitHub issue #270: mloda 0.9.0 rejects HashableDict in
credential / BaseInputData data-access construction. Detection is marker-anchored
and bracket-span based:

- A construction-shaped credential marker is one of ``credentials=``,
  ``add_credentials(``, or a quoted ``"BaseInputData"`` group key.
- From each marker match, walk forward character by character tracking bracket
  depth. The marker's construct span ends when depth goes negative (the enclosing
  construct was exited) or when a newline is seen at depth 0 (the statement ended),
  with a safety cap so an unclosed snippet cannot run away.
- Any ``HashableDict`` token whose offset falls inside such a span is flagged, at
  the token's 1-based line, deduped to one error per line.

Legitimate HashableDict usage (Options group values, ApiInputDataCollection,
provider export/import) is left untouched because it is not inside a marker span.

Known limitations (accepted trade-offs, matched against the literal source text):
- Aliased imports evade detection. ``from mloda.provider import HashableDict as HD``
  followed by ``HD(...)`` is not caught: the guard matches the literal
  ``HashableDict`` token only.
- No dataflow / variable-indirection analysis. ``credentials=creds`` where
  ``creds = HashableDict(...)`` is assigned elsewhere is not caught: detection is
  limited to ``HashableDict`` appearing inside the credential construct's own
  brackets.
- Only the quoted ``"BaseInputData"`` group-key form is matched (the documented
  reader-tuple form). An unquoted ``BaseInputData`` class reference is intentionally
  not treated as a marker, to avoid false positives on that ubiquitous token.

Run: python scripts/lint_credentials.py
Exit code: 1 if any issues found, 0 otherwise.
"""

import os
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

CREDENTIAL_MARKER_RE = re.compile(r"\bcredentials\s*=|\badd_credentials\s*\(|[\"']BaseInputData[\"']")

# Safety cap: stop scanning a single marker's span after this many lines so a
# malformed / unclosed snippet cannot run away.
_MAX_SPAN_LINES = 50


def _marker_span_end(content: str, start: int) -> int:
    """Return the exclusive end offset of the bracket-balanced span from ``start``.

    Walk forward tracking bracket depth. Stop when depth goes negative (the
    enclosing construct was exited), when a newline is seen at depth 0 (the
    statement ended), or when the safety cap of scanned lines is reached.
    """
    depth = 0
    newlines = 0
    for i in range(start, len(content)):
        ch = content[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                return i
        elif ch == "\n":
            newlines += 1
            if depth == 0 or newlines >= _MAX_SPAN_LINES:
                return i
    return len(content)


def scan_content(rel_path: str, content: str) -> list[str]:
    """Flag each HashableDict token that falls inside a credential-marker span.

    Pure: takes text, returns violation strings. Each violation contains the
    substrings ``HashableDict`` and ``credential`` plus a ``{rel_path}:{lineno}:``
    locator, where lineno is the 1-based line of the HashableDict token. Lines are
    deduped, so a token inside multiple overlapping spans yields a single error.
    """
    spans = [(m.end(), _marker_span_end(content, m.end())) for m in CREDENTIAL_MARKER_RE.finditer(content)]
    flagged_lines: set[int] = set()
    for match in HASHABLE_DICT_RE.finditer(content):
        offset = match.start()
        if any(start <= offset < end for start, end in spans):
            flagged_lines.add(content.count("\n", 0, offset) + 1)
    return [
        f"{rel_path}:{lineno}: HashableDict used in credential / data-access construction"
        for lineno in sorted(flagged_lines)
    ]


def _is_excluded_dir(part: str) -> bool:
    return part in EXCLUDE_DIRS or part.endswith(".egg-info")


def find_credential_hashable_dict(root: Path) -> list[str]:
    """Walk root for .py/.md files and return sorted credential-context HashableDict violations."""
    errors: list[str] = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not _is_excluded_dir(d)]
        for name in files:
            path = Path(dirpath) / name
            if path.suffix not in SCAN_SUFFIXES:
                continue
            rel_posix = path.relative_to(root).as_posix()
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
