"""Verify that compute-framework packages required for testing are installed.

This prevents silent test skipping where pytest.importorskip() causes entire
test modules to be collected but skipped without any CI failure signal.
See: https://github.com/mloda-ai/mloda-registry/issues/135
"""

from __future__ import annotations

import importlib

import pytest


REQUIRED_TEST_PACKAGES = [
    "pandas",
    "polars",
    "duckdb",
]


@pytest.mark.parametrize("package", REQUIRED_TEST_PACKAGES)
def test_required_test_dependency_installed(package: str) -> None:
    """Each compute-framework package used in tests must be importable.

    If this test fails, the package is missing from the dev dependencies
    in the root pyproject.toml. Add it to [project.optional-dependencies] dev.
    """
    try:
        importlib.import_module(package)
    except ImportError:
        pytest.fail(
            f"Required test dependency '{package}' is not installed. "
            f"Add it to [project.optional-dependencies] dev in pyproject.toml."
        )
