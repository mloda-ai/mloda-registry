"""Tests that verify all expected optional test dependencies are installed.

These tests prevent silent test skipping: if a dependency listed here is
missing from the dev extras in pyproject.toml, the test suite will fail
loudly instead of silently skipping entire test modules via
``pytest.importorskip``.
"""

import importlib

import pytest

# Every framework whose test files use ``pytest.importorskip`` must appear
# here.  Adding a new compute-framework backend with importorskip-guarded
# tests?  Add its import name to this list so CI catches missing deps early.
REQUIRED_TEST_DEPENDENCIES = [
    "duckdb",
    "pandas",
    "polars",
    "pyarrow",
]


@pytest.mark.parametrize("module_name", REQUIRED_TEST_DEPENDENCIES)
def test_required_test_dependency_is_installed(module_name: str) -> None:
    """Each framework dependency must be importable in the test environment.

    If this test fails, add the missing package to the ``dev`` extras in
    the root ``pyproject.toml``.
    """
    try:
        importlib.import_module(module_name)
    except ImportError:
        pytest.fail(
            f"Required test dependency '{module_name}' is not installed. "
            f"Add it to [project.optional-dependencies] dev in the root pyproject.toml."
        )
