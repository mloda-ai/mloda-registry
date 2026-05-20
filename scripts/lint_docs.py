"""Lint documentation guides for broken relative links and internal imports in code snippets.

Run: python scripts/lint_docs.py
Exit code: 1 if any issues found, 0 otherwise.
"""

import functools
import re
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs" / "guides"

INTERNAL_IMPORT_RE = re.compile(r"from mloda\.core\.")

RELATIVE_LINK_RE = re.compile(r"\[.*?\]\((?!https?://|mailto:)([^)#\s]+\.md)(?:#([^)\s]+))?\)")

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

CODE_BLOCK_RE = re.compile(r"^```", re.MULTILINE)


def find_code_blocks(content: str) -> list[str]:
    """Extract code block contents from markdown."""
    blocks = []
    parts = CODE_BLOCK_RE.split(content)
    for i in range(1, len(parts), 2):
        blocks.append(parts[i])
    return blocks


def _slugify_heading(text: str) -> str:
    """Simplified GFM-style slug: lowercase, strip backticks, spaces->hyphens, drop non-[\\w-]. Does NOT handle duplicate-heading disambiguation (`-1`, `-2`) or stripping of inline markdown formatting beyond backticks."""
    text = text.lower().replace("`", "")
    text = text.replace(" ", "-")
    return re.sub(r"[^\w-]", "", text)


@functools.lru_cache(maxsize=None)
def _heading_slugs(md_file: Path) -> frozenset[str]:
    slugs: set[str] = set()
    for match in HEADING_RE.finditer(md_file.read_text()):
        slugs.add(_slugify_heading(match.group(2)))
    return frozenset(slugs)


def check_relative_links_and_anchors(md_file: Path, content: str) -> list[str]:
    """Validate relative markdown links and their optional anchor fragments."""
    errors = []
    for match in RELATIVE_LINK_RE.finditer(content):
        rel_path = match.group(1)
        anchor = match.group(2)
        target = (md_file.parent / rel_path).resolve()
        if not target.exists():
            line_num = content[: match.start()].count("\n") + 1
            errors.append(f"{md_file}:{line_num}: broken link -> {rel_path}")
            continue
        if anchor and target.is_file():
            slug = _slugify_heading(anchor)
            if slug not in _heading_slugs(target):
                line_num = content[: match.start()].count("\n") + 1
                errors.append(f"{md_file}:{line_num}: broken anchor -> {rel_path}#{anchor}")
    return errors


def check_internal_imports(md_file: Path, content: str) -> list[str]:
    errors = []
    code_blocks = find_code_blocks(content)
    for block in code_blocks:
        for line in block.splitlines():
            stripped = line.strip()
            if INTERNAL_IMPORT_RE.search(stripped):
                line_num = content.find(stripped)
                if line_num >= 0:
                    line_num = content[:line_num].count("\n") + 1
                errors.append(f"{md_file}:{line_num}: internal import in code snippet -> {stripped}")
    return errors


def main() -> int:
    if not DOCS_DIR.is_dir():
        print(f"Docs directory not found: {DOCS_DIR}")
        return 1

    all_errors: list[str] = []

    for md_file in sorted(DOCS_DIR.rglob("*.md")):
        content = md_file.read_text()
        all_errors.extend(check_relative_links_and_anchors(md_file, content))
        all_errors.extend(check_internal_imports(md_file, content))

    if all_errors:
        print(f"Found {len(all_errors)} doc issue(s):\n")
        for error in all_errors:
            print(f"  {error}")
        return 1

    print("All docs OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
