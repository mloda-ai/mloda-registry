"""Doc-drift guard for ``docs/guides/feature-group-patterns/28-binary-backed-features.md``.

There is no mktestdocs/sybil-style execution of guide code fences in this repo (confirmed: no such
tool is wired in), so this is a lightweight grep-based check instead of an executed doctest.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GUIDE_PATH = _REPO_ROOT / "docs" / "guides" / "feature-group-patterns" / "28-binary-backed-features.md"


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
