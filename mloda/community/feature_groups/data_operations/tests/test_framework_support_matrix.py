"""Drift check: the framework support matrix doc must match supported_*() sets.

If this test fails, regenerate the doc:

    python scripts/check_framework_support_matrix.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[5]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_framework_support_matrix.py"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_check_framework_support_matrix", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_framework_support_matrix_is_in_sync() -> None:
    mod = _load_script_module()
    collected = [mod.collect_operation(op) for op in mod.OPERATIONS]
    generated = mod.render_generated_block(collected)

    current = mod.DOC_PATH.read_text()
    expected = mod.splice_into_doc(current, generated)

    assert expected == current, (
        "framework-support-matrix.md is out of sync with supported_*() sets.\n"
        "Run: python scripts/check_framework_support_matrix.py"
    )


def test_operations_list_covers_every_data_operation_on_disk() -> None:
    mod = _load_script_module()
    uncovered = mod.discover_uncovered_tests_packages()
    assert uncovered == [], (
        "OPERATIONS in scripts/check_framework_support_matrix.py is missing entries "
        "for these tests packages (each has test_<framework>.py files):\n  " + "\n  ".join(uncovered)
    )
