"""Real-wheel system test: exercises the ``example_binary`` distribution (``mloda-example-binary``
on PyPI/TestPyPI) actually installed, instead of the simulated binary from
``mloda-testing[binary-model]``. Skipped by default -- the wheel is never installed in CI; see
``docs/guides/feature-group-patterns/28-binary-backed-features.md``.

The full expired/in-grace/valid license state machine is already covered against the real compiled
binary in the mloda-binary-wrapper repo's own CI across multiple platforms, and is deliberately not
duplicated here. A CI job that installs the real wheel in this repo, and the full production-license
end-to-end path, are intentionally deferred follow-up work, not omissions.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import pyarrow as pa
import pytest

example_binary = pytest.importorskip("example_binary")

from mloda.community.feature_groups.binary_model.binary import CONTRACT_VERSION
from mloda.community.feature_groups.binary_model.errors import LicenseInvalidError, LicenseMissingError
from mloda.enterprise.feature_groups.binary_example.binary_example_feature_group import BinaryExampleFeatureGroup
from mloda.provider import ApiInputDataFeature, FeatureSet
from mloda.testing.binary_model.conformance import run_binary, write_json
from mloda.testing.binary_model.hash_reference import compute_expected_hash_column
from mloda.testing.binary_model.license_vectors import valid_license_token
from mloda.user import Feature, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.test_binary_model_real.probe_classification import UNKNOWN_TEST_KEY_MESSAGE, classify_test_key_probe

_BINARY_PATH: Path = example_binary.binary_path()
_PLUGIN_ID = BinaryExampleFeatureGroup.BINARY_PLUGIN_ID


def _probe_accepts_test_key(binary_path: Path) -> bool:
    """True if ``binary_path`` accepts the shared test-signed vectors from ``license_vectors``;
    False for a release build, which trusts only ``PRODUCTION_KEYS`` (currently empty, so a
    test-signed token is always an unknown ``kid``). Delegates exit-code/message interpretation to
    ``classify_test_key_probe``, which raises loudly for anything else."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = write_json(
            Path(tmp_dir) / "config.json",
            {"input_columns": ["col_a"], "operation": "hash", "parameters": {}, "output_columns": {"result": "out"}},
        )
        env = {"MLODA_LICENSE_KEY": valid_license_token([_PLUGIN_ID])}
        result = run_binary([str(binary_path)], ["run", "--config", str(config_path)], env)
    return classify_test_key_probe(result)


_ACCEPTS_TEST_KEY = _probe_accepts_test_key(_BINARY_PATH)


def _single_hash_feature(columns: list[str]) -> tuple[Feature, FeatureSet]:
    feature = Feature("hashed", Options(context={"binary_operation": "hash", "binary_input_columns": columns}))
    feature_set = FeatureSet()
    feature_set.add(feature)
    return feature, feature_set


class _RealWheelTestLicense(BinaryExampleFeatureGroup):
    """Production class, real wheel (no BINARY_COMMAND_OVERRIDE), shared test-signed license."""

    LICENSE_KEY_OVERRIDE = valid_license_token([_PLUGIN_ID])


# -------------------------------------------------------------------------------------------
# Tests that must pass against either kind of installed wheel (release or test-key build)
# -------------------------------------------------------------------------------------------


def test_version_probe_succeeds_and_matches_plugin_id_and_semver() -> None:
    result = run_binary([str(_BINARY_PATH)], ["--version"], {})
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    line = result.stdout.decode("utf-8").strip()
    assert re.fullmatch(rf"{re.escape(_PLUGIN_ID)} \d+\.\d+\.\d+(?:[-+][0-9A-Za-z.+-]+)?", line), line


def test_capabilities_reports_contract_and_plugin_id() -> None:
    result = run_binary([str(_BINARY_PATH)], ["--capabilities"], {})
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    payload = json.loads(result.stdout.decode("utf-8").strip())
    assert payload.get("contract") == CONTRACT_VERSION
    assert payload.get("plugin_id") == _PLUGIN_ID


def test_no_license_raises_license_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production class, real binary_path(), no BINARY_COMMAND_OVERRIDE, no license source set."""
    monkeypatch.delenv("MLODA_LICENSE_FILE", raising=False)
    monkeypatch.delenv("MLODA_LICENSE_KEY", raising=False)
    table = pa.table({"col_a": ["alpha"]})
    _, feature_set = _single_hash_feature(["col_a"])
    with pytest.raises(LicenseMissingError):
        BinaryExampleFeatureGroup.calculate_feature(table, feature_set)


def test_test_signed_license_is_rejected_or_accepted_depending_on_the_installed_build() -> None:
    """A release build rejects the shared test-signed token as an unknown key id; a test-key
    build accepts it."""
    rows: dict[str, list[str]] = {"col_a": ["alpha", "beta"]}
    table = pa.table(rows)
    _, feature_set = _single_hash_feature(["col_a"])
    if _ACCEPTS_TEST_KEY:
        result = _RealWheelTestLicense.calculate_feature(table, feature_set)
        assert result.column("hashed").to_pylist() == compute_expected_hash_column(rows, ["col_a"], None)
    else:
        with pytest.raises(LicenseInvalidError, match=UNKNOWN_TEST_KEY_MESSAGE):
            _RealWheelTestLicense.calculate_feature(table, feature_set)


# -------------------------------------------------------------------------------------------
# Test-key-build-only: a full end-to-end run needs a license the real binary actually accepts
# -------------------------------------------------------------------------------------------


@pytest.mark.skipif(
    not _ACCEPTS_TEST_KEY,
    reason=(
        "test-key build not installed: this real wheel rejects the shared test-signed license "
        "vectors as unknown-kid (PRODUCTION_KEYS is currently empty), so no license from "
        "license_vectors can drive a full run"
    ),
)
def test_real_binary_end_to_end_with_valid_test_license() -> None:
    rows: dict[str, list[str]] = {"col_a": ["alpha", "beta", "gamma"]}
    feature, _ = _single_hash_feature(["col_a"])
    results = mloda.run_all(
        [feature],
        compute_frameworks={PyArrowTable},
        api_data={"BinaryExampleRealWheelData": rows},
        plugin_collector=PluginCollector.enabled_feature_groups({ApiInputDataFeature, _RealWheelTestLicense}),
    )
    expected = compute_expected_hash_column(rows, ["col_a"], None)
    found = False
    for table in results:
        if isinstance(table, pa.Table) and "hashed" in table.column_names:
            assert table.column("hashed").to_pylist() == expected
            found = True
    assert found, "hashed column not found in any result table"
