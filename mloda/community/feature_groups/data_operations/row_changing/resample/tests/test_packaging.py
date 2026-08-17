"""Packaging lint: resample's backend modules must not import from time_bucketization.

config/packages.toml declares only mloda-community-data-operations as resample's
dependency, never mloda-community-time-bucketization. A pip install of
mloda-community-resample on its own never pulls in time_bucketization, so any
cross-package import from it in a resample backend module breaks at plugin-discovery
time in a real install, invisible to the checkout-local test suite (which always has
the whole monorepo on sys.path). ``time_bucketization`` never appears in these files
for any other reason, so a bare substring search is both necessary and sufficient.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_RESAMPLE_DIR = Path(__file__).resolve().parent.parent

_FORBIDDEN_SUBSTRING = "time_bucketization"


@pytest.mark.parametrize(
    "filename",
    [
        pytest.param("duckdb_resample.py", id="duckdb"),
        pytest.param("pandas_resample.py", id="pandas"),
        pytest.param("polars_lazy_resample.py", id="polars_lazy"),
    ],
)
def test_backend_does_not_import_time_bucketization(filename: str) -> None:
    source = (_RESAMPLE_DIR / filename).read_text()
    assert _FORBIDDEN_SUBSTRING not in source, (
        f"{filename} contains {_FORBIDDEN_SUBSTRING!r}; resample's own config/packages.toml entry "
        "declares no dependency on mloda-community-time-bucketization, so a resample backend module "
        "must not import from it (the shared helpers belong in the data_operations base package)."
    )
