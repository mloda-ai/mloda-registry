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


# Maps a framework name to one representative test module that uses
# ``pytest.importorskip`` at module level.  If the dependency is missing,
# importing the module raises ``pytest.skip.Exception``.
_FRAMEWORK_TEST_MODULES = {
    "duckdb": "mloda.community.feature_groups.data_operations.aggregation.tests.test_duckdb",
    "pandas": "mloda.community.feature_groups.data_operations.aggregation.tests.test_pandas",
    "polars": "mloda.community.feature_groups.data_operations.aggregation.tests.test_polars_lazy",
}


@pytest.mark.parametrize(
    ("framework", "module_path"),
    _FRAMEWORK_TEST_MODULES.items(),
    ids=_FRAMEWORK_TEST_MODULES.keys(),
)
def test_framework_tests_are_not_skipped(framework: str, module_path: str) -> None:
    """Importing a framework test module must not trigger pytest.skip.

    When a dependency is missing, the module-level ``pytest.importorskip``
    raises ``pytest.skip.Exception`` at import time, causing every test in
    that file to be silently skipped.  This test imports the module directly
    and fails if the skip fires, proving the tests actually run.
    """
    try:
        importlib.import_module(module_path)
    except pytest.skip.Exception:
        pytest.fail(
            f"{framework} test module was skipped at import time. "
            f"{framework} is not installed; add it to [project.optional-dependencies] dev."
        )
