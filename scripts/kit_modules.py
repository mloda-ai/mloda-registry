"""Shared map from every shipped mloda-testing module to the extra that gates it (None = ungated)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KIT_PATH = "mloda/testing"

# Kit-relative posix path prefix -> mloda-testing extra; the longest matching prefix wins.
KIT_GATES: dict[str, str] = {
    "binary_model/": "binary-model",
    "extenders/": "pyarrow",
    "extenders/otel.py": "otel",
    "extenders/openlineage.py": "openlineage",
}

_SKIPPED_PARTS = {"tests", "__pycache__", "build", "dist"}


def kit_gate(rel_posix: str) -> str | None:
    """The extra gating a kit-relative posix path, or None when ungated."""
    matches = [prefix for prefix in KIT_GATES if rel_posix.startswith(prefix)]
    return KIT_GATES[max(matches, key=len)] if matches else None


def shipped_modules(root: Path, package_path: str) -> dict[str, str]:
    """Root-relative posix path to dotted module name for every shipped module under root."""
    modules: dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if any(part in _SKIPPED_PARTS or part.endswith(".egg-info") for part in rel.parts):
            continue
        parts = [*package_path.split("/"), *rel.with_suffix("").parts]
        if parts[-1] == "__init__":
            parts.pop()
        modules[rel.as_posix()] = ".".join(parts)
    return modules


def kit_modules(root: Path = REPO_ROOT / KIT_PATH) -> dict[str | None, list[str]]:
    """Sorted dotted names of every shipped kit module under root, grouped by gating extra."""
    grouped: dict[str | None, list[str]] = {}
    for rel, name in shipped_modules(root, KIT_PATH).items():
        grouped.setdefault(kit_gate(rel), []).append(name)
    return {gate: sorted(modules) for gate, modules in grouped.items()}
