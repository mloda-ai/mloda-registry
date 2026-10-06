"""Real-wheel system test: exercises the ``example_binary`` (``mloda-example-binary``) and
``anonymizer_binary`` (``mloda-anonymizer-binary``) distributions on PyPI actually installed, instead of
the simulated binary from ``mloda-testing[binary-model]``. Each case is skipped unless its wheel is
installed; the ``real-wheel`` CI job (``tox -e real-wheel``) installs the release wheels from the lock. See
``docs/guides/feature-group-patterns/29-binary-backed-features.md``.

Run this directory on its own with ``pytest tests/test_binary_model_real/``; it needs no opt-in. The
repo's other suites assume the wheel is absent and are not supported with it installed.

The full expired/in-grace/valid license state machine is already covered against the real compiled
binary in the mloda-binary-wrapper repo's own CI across multiple platforms, and is deliberately not
duplicated here. The end-to-end test needs a test-key build (never published), so it skips against the
release wheel. That skip is deliberate: CI holds no production license and no access to the private wrapper
repo, so no secret can leak; a production-license run is a manual check at key issuance. The ``real-wheel`` CI
job runs this directory on Linux x86_64, macOS and Windows, not Linux aarch64 (the wrapper does not
run-test it). The manual production-license run is
``MLODA_LICENSE_FILE=<license> pytest tests/test_binary_model_real/``.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from mloda.provider import ApiInputDataFeature, FeatureSet
from mloda.user import Feature, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.binary_model.contract import CONTRACT_VERSION
from mloda.community.feature_groups.binary_model.errors import LicenseInvalidError, LicenseMissingError
from mloda.enterprise.feature_groups.anonymizer.anonymizer_feature_group import AnonymizerFeatureGroup
from mloda.enterprise.feature_groups.binary_example.binary_example_feature_group import BinaryExampleFeatureGroup
from mloda.testing.binary_model import VERSION_PATTERN
from mloda.testing.binary_model.conformance import run_binary
from mloda.testing.binary_model.hash_reference import compute_expected_hash_column
from mloda.testing.binary_model.hmac_sha256_reference import KNOWN_ANSWER_KEY, compute_expected_hmac_sha256_column
from mloda.testing.binary_model.license_vectors import valid_license_token
from tests.test_binary_model_real.probe_classification import (
    UNKNOWN_TEST_KEY_MESSAGE,
    probe_accepts_test_key,
    probe_environment,
)

_ANONYMIZER_KEY_ENV = "REAL_WHEEL_ANONYMIZER_PII_KEY"


class _ExampleTestLicense(BinaryExampleFeatureGroup):
    """Production class, real wheel (no BINARY_COMMAND_OVERRIDE), shared test-signed license."""

    LICENSE_KEY_OVERRIDE = valid_license_token([BinaryExampleFeatureGroup.BINARY_PLUGIN_ID])


class _AnonymizerTestLicense(AnonymizerFeatureGroup):
    """Production class, real wheel (no BINARY_COMMAND_OVERRIDE), shared test-signed license."""

    LICENSE_KEY_OVERRIDE = valid_license_token([AnonymizerFeatureGroup.BINARY_PLUGIN_ID])


@dataclass(frozen=True)
class _Case:
    """One real wheel: its import module, production and test-licensed classes, feature and expected values."""

    module: str
    production_class: type[BinaryExampleFeatureGroup] | type[AnonymizerFeatureGroup]
    test_license_class: type[BinaryExampleFeatureGroup] | type[AnonymizerFeatureGroup]
    column: str
    source_column: str
    make_feature: Callable[[], Feature]
    expected: Callable[[dict[str, list[str]]], list[Any]]
    probe_kwargs: dict[str, Any]


_CASES: dict[str, _Case] = {
    "example": _Case(
        module="example_binary",
        production_class=BinaryExampleFeatureGroup,
        test_license_class=_ExampleTestLicense,
        column="hashed",
        source_column="col_a",
        make_feature=lambda: Feature(
            "hashed", Options(context={"binary_operation": "hash", "binary_input_columns": ["col_a"]})
        ),
        expected=lambda rows: compute_expected_hash_column(rows, ["col_a"], None),
        probe_kwargs={},
    ),
    "anonymizer": _Case(
        module="anonymizer_binary",
        production_class=AnonymizerFeatureGroup,
        test_license_class=_AnonymizerTestLicense,
        column="pseudonymized",
        source_column="email",
        make_feature=lambda: Feature(
            "pseudonymized",
            Options(
                context={
                    "pseudonymization_algorithm": "hmac_sha256",
                    "pii_key_env": _ANONYMIZER_KEY_ENV,
                    "in_features": "email",
                }
            ),
        ),
        expected=lambda rows: compute_expected_hmac_sha256_column(rows["email"], KNOWN_ANSWER_KEY),
        probe_kwargs={"operation": "hmac_sha256", "parameters": {"key": KNOWN_ANSWER_KEY}},
    ),
}


@dataclass(frozen=True)
class _RealCase:
    case: _Case
    binary_path: Path
    plugin_id: str
    accepts_test_key: Callable[[], bool]


_accepts_test_key: dict[str, bool] = {}


@pytest.fixture(params=list(_CASES))
def real_case(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> _RealCase:
    """Skips unless the case's wheel is installed; sets a valid anonymizer key so a missing key never
    masks the license error a test asserts."""
    case = _CASES[request.param]
    module = pytest.importorskip(case.module)
    binary_path: Path = module.binary_path()
    monkeypatch.setenv(_ANONYMIZER_KEY_ENV, KNOWN_ANSWER_KEY)

    def accepts_test_key() -> bool:
        if case.module not in _accepts_test_key:
            _accepts_test_key[case.module] = probe_accepts_test_key(
                [str(binary_path)], plugin_id=case.production_class.BINARY_PLUGIN_ID, **case.probe_kwargs
            )
        return _accepts_test_key[case.module]

    return _RealCase(case, binary_path, case.production_class.BINARY_PLUGIN_ID, accepts_test_key)


def _single_feature(case: _Case) -> tuple[Feature, FeatureSet]:
    feature = case.make_feature()
    feature_set = FeatureSet()
    feature_set.add(feature)
    return feature, feature_set


# -------------------------------------------------------------------------------------------
# Tests that must pass against either kind of installed wheel (release or test-key build)
# -------------------------------------------------------------------------------------------


def test_version_probe_succeeds_and_matches_plugin_id_and_semver(real_case: _RealCase) -> None:
    result = run_binary([str(real_case.binary_path)], ["--version"], probe_environment())
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    line = result.stdout.decode("utf-8").strip()
    assert re.fullmatch(rf"{re.escape(real_case.plugin_id)} {VERSION_PATTERN}", line), line


def test_capabilities_reports_contract_and_plugin_id(real_case: _RealCase) -> None:
    result = run_binary([str(real_case.binary_path)], ["--capabilities"], probe_environment())
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    payload = json.loads(result.stdout.decode("utf-8").strip())
    assert payload.get("contract") == CONTRACT_VERSION
    assert payload.get("plugin_id") == real_case.plugin_id


def test_no_license_raises_license_missing(real_case: _RealCase, monkeypatch: pytest.MonkeyPatch) -> None:
    """Production class, real binary_path(), no BINARY_COMMAND_OVERRIDE, no license source set."""
    monkeypatch.delenv("MLODA_LICENSE_FILE", raising=False)
    monkeypatch.delenv("MLODA_LICENSE_KEY", raising=False)
    case = real_case.case
    table = pa.table({case.source_column: ["alpha"]})
    _, feature_set = _single_feature(case)
    with pytest.raises(LicenseMissingError):
        case.production_class.calculate_feature(table, feature_set)


def test_test_signed_license_is_rejected_or_accepted_depending_on_the_installed_build(real_case: _RealCase) -> None:
    """A release build rejects the shared test-signed token as an unknown key id; a test-key
    build accepts it."""
    case = real_case.case
    rows: dict[str, list[str]] = {case.source_column: ["alpha", "beta"]}
    table = pa.table(rows)
    _, feature_set = _single_feature(case)
    if real_case.accepts_test_key():
        result = case.test_license_class.calculate_feature(table, feature_set)
        assert result.column(case.column).to_pylist() == case.expected(rows)
    else:
        with pytest.raises(LicenseInvalidError, match=UNKNOWN_TEST_KEY_MESSAGE):
            case.test_license_class.calculate_feature(table, feature_set)


# -------------------------------------------------------------------------------------------
# Test-key-build-only: a full end-to-end run needs a license the real binary actually accepts
# -------------------------------------------------------------------------------------------


def _run_end_to_end(case: _Case, feature_class: type[BinaryExampleFeatureGroup] | type[AnonymizerFeatureGroup]) -> None:
    rows: dict[str, list[str]] = {case.source_column: ["alpha", "beta", "gamma"]}
    feature, _ = _single_feature(case)
    results = mloda.run_all(
        [feature],
        compute_frameworks=[PyArrowTable],
        api_data={"BinaryRealWheelData": rows},
        plugin_collector=PluginCollector.enabled_feature_groups({ApiInputDataFeature, feature_class}),
    )
    expected = case.expected(rows)
    found = False
    for table in results:
        if isinstance(table, pa.Table) and case.column in table.column_names:
            assert table.column(case.column).to_pylist() == expected
            found = True
    assert found, f"{case.column} column not found in any result table"


@pytest.mark.parametrize("case_name", list(_CASES))
def test_test_license_class_ignores_ambient_license_file(case_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit test key stops the ambient MLODA_LICENSE_FILE being forwarded, which the binary reads first."""
    monkeypatch.setenv("MLODA_LICENSE_FILE", "/ambient/license.txt")
    case = _CASES[case_name]
    env = case.test_license_class.binary_environment()
    assert "MLODA_LICENSE_FILE" not in env
    assert env["MLODA_LICENSE_KEY"] == case.test_license_class.LICENSE_KEY_OVERRIDE


def test_real_binary_end_to_end_with_valid_test_license(real_case: _RealCase) -> None:
    if not real_case.accepts_test_key():
        pytest.skip(
            "test-key build not installed: this real wheel is a release build, which trusts only "
            "production keys and rejects the shared test-signed license vectors as unknown-kid, so no "
            "license from license_vectors can drive a full run"
        )
    _run_end_to_end(real_case.case, real_case.case.test_license_class)


def test_real_binary_end_to_end_with_production_license(real_case: _RealCase) -> None:
    """Manual production-license run: the caller's MLODA_LICENSE_FILE is forwarded to the real binary."""
    if not os.environ.get("MLODA_LICENSE_FILE"):
        pytest.skip(
            "manual production-license run: set MLODA_LICENSE_FILE to a production license entitled to this wheel"
        )
    _run_end_to_end(real_case.case, real_case.case.production_class)
