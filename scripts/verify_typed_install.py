#!/usr/bin/env python3
"""Prove a standalone leaf install is typed: run mypy --strict over a probe against the released wheels.

Run: python scripts/verify_typed_install.py <version>
Exit code: 1 if the captured mypy output does not prove the install typed, 0 otherwise. Only the output
counts: against an install missing py.typed, mypy sees Any everywhere and still exits 0.
"""

from __future__ import annotations

import argparse
import subprocess  # nosec
import sys
import tempfile
from pathlib import Path

# The leaf under probe plus the base distribution that ships the py.typed marker.
PROBE_DISTRIBUTIONS = ("mloda-community-aggregation", "mloda-community-data-operations")

# Line 4 reveals the attribute's type; line 5 provokes [call-arg] and [assignment] on a typed install.
PROBE_SOURCE = '''\
"""Probe: a typed install reveals a concrete signature and yields both provoked errors."""
from mloda.community.feature_groups.data_operations.aggregation.pyarrow_aggregation import PyArrowAggregation

reveal_type(PyArrowAggregation.supported_subtypes)
BAD: int = PyArrowAggregation.supported_subtypes(secondary=123)
'''


def mypy_output_problems(output: str) -> list[str]:
    """Reasons the captured mypy output does not prove the install typed; empty means proven."""
    problems: list[str] = []
    if 'Revealed type is "Any"' in output:
        problems.append("the probe import resolved to Any: 'Revealed type is \"Any\"' (py.typed marker missing?)")
    elif "Revealed type is" not in output:
        problems.append("no 'Revealed type is' line: the probe never ran")
    if 'Unexpected keyword argument "secondary"' not in output:
        problems.append("missing the provoked 'Unexpected keyword argument \"secondary\"' [call-arg] error")
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
        venv = Path(tmpdir) / "venv"
        # Not verify_build_floor.venv_python: importing it would drag a toml parser into this env.
        python = venv / "bin" / "python"
        pins = [f"{name}=={args.version}" for name in PROBE_DISTRIBUTIONS]
        commands = [
            ["uv", "venv", "--python", sys.executable, str(venv)],
            ["uv", "pip", "install", "--python", str(python), *pins, "mypy"],
        ]
        for command in commands:
            # cwd is the temp dir, so the workspace [tool.uv] config cannot filter the install.
            result = subprocess.run(command, capture_output=True, text=True, cwd=tmpdir)  # nosec
            if result.returncode != 0:
                print(f"❌ {' '.join(command)} failed:\n{result.stderr[-500:]}")
                return 1

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
