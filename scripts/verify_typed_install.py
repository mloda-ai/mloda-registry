#!/usr/bin/env python3
"""Prove a standalone leaf install is typed: run mypy --strict over a probe against the released wheels.

Run: python scripts/verify_typed_install.py <version>
Exit code: 1 if the captured mypy output does not prove the install typed, 0 otherwise. Only the output
counts: against an install missing py.typed, mypy sees Any everywhere and still exits 0.
"""

from __future__ import annotations

import argparse
import runpy
import subprocess  # nosec
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent

# Not a plain import: this script is also loaded by file path in tests, where scripts/ is not on sys.path.
_load_sibling: Callable[[str], ModuleType] = runpy.run_path(str(REPO_ROOT / "scripts" / "script_loader.py"))[
    "load_sibling"
]
_common = _load_sibling("verify_common")
install: Callable[[list[str], str], str | None] = _common.install
venv_dir: Callable[[str], Path] = _common.venv_dir
venv_python: Callable[[Path], Path] = _common.venv_python

# The leaf under probe plus the base distribution that ships the py.typed marker.
PROBE_DISTRIBUTIONS = ("mloda-community-aggregation", "mloda-community-data-operations")

# reveal_type shows the leaf-defined classmethod; the last two lines provoke [assignment] and [arg-type] on a typed install.
PROBE_SOURCE = '''\
"""Probe: a typed install reveals a concrete signature and yields both provoked errors."""
from mloda.provider import FeatureSet
from mloda.community.feature_groups.data_operations.aggregation.pyarrow_aggregation import PyArrowAggregation

reveal_type(PyArrowAggregation.calculate)
BAD: int = PyArrowAggregation.supported
PyArrowAggregation.calculate(123, [], FeatureSet(), None)
'''


def mypy_output_problems(output: str) -> list[str]:
    """Reasons the captured mypy output does not prove the install typed; empty means proven."""
    problems: list[str] = []
    if 'Revealed type is "Any"' in output:
        problems.append("the probe import resolved to Any: 'Revealed type is \"Any\"' (py.typed marker missing?)")
    elif "Revealed type is" not in output:
        problems.append("no 'Revealed type is' line: the probe never ran")
    if "[arg-type]" not in output:
        problems.append(
            "missing the provoked '[arg-type]' error on the calculate(123, ...) call (int vs type[AggregationFeatureGroup])"
        )
    if "[assignment]" not in output:
        problems.append("missing the provoked '[assignment]' error")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a standalone leaf install is typed under mypy --strict")
    parser.add_argument("version", nargs="?", default="", help="Released version to install the probe pair at")
    args = parser.parse_args()

    # tox renders an empty argument when MLODA_REGISTRY_VERSION is unset.
    if not args.version.strip():
        parser.error("version must not be empty (is MLODA_REGISTRY_VERSION set?)")

    with tempfile.TemporaryDirectory() as tmpdir:
        pins = [f"{name}=={args.version}" for name in PROBE_DISTRIBUTIONS]
        error = install([*pins, "mypy"], tmpdir)
        if error is not None:
            print(f"❌ {error}")
            return 1

        python = venv_python(venv_dir(tmpdir))
        (Path(tmpdir) / "probe.py").write_text(PROBE_SOURCE)
        # cwd is the temp dir, so the checkout cannot shadow the installed packages.
        result = subprocess.run(  # nosec
            [str(python), "-m", "mypy", "--strict", "--ignore-missing-imports", "probe.py"],
            capture_output=True,
            text=True,
            cwd=tmpdir,
        )

    output = result.stdout + result.stderr
    problems = mypy_output_problems(output)
    if problems:
        print(output)
        print("❌ Problems:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"✅ {PROBE_DISTRIBUTIONS[0]}=={args.version} installs typed under mypy --strict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
