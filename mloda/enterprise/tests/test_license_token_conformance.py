"""Runs ``LicenseTokenConformanceMixin`` against the licensed binary in
``mloda.enterprise.licensing.testing.licensed_binary``, plus the token hooks' retargeting, the in-process
``check_license_token``, the light flags of both licensed binaries and the licensed hmac binary.
Needs ``mloda-testing[binary-model]`` and ``mloda-enterprise[ed25519]``."""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock

import pyarrow as pa
import pytest

from mloda.community.feature_groups.binary_model.transport import minimal_environment
from mloda.enterprise.licensing.testing.conformance import LicenseTokenConformanceMixin
from mloda.enterprise.licensing.testing.license_vectors import (
    expired_license_token,
    expiring_license_token,
    in_grace_license_token,
    unknown_kid_license_token,
    valid_license_token,
)
from mloda.enterprise.licensing.testing.licensed_binary import check_license_token
from mloda.testing.binary_model import hmac_sha256_reference
from mloda.testing.binary_model.conformance import (
    LICENSE_INVALID,
    arrow_stream_bytes,
    read_arrow_stream,
    run_binary,
    write_json,
)
from mloda.testing.binary_model.simulated_binary import FakeLicenseError, simulated_license
from mloda.testing.tests._module_probe import run_module_probe

_LICENSE_KEY = "MLODA_LICENSE_KEY"
_LICENSE_FILE = "MLODA_LICENSE_FILE"

_LICENSED_BINARY_MODULE = "mloda.enterprise.licensing.testing.licensed_binary"
_LICENSED_HMAC_BINARY_MODULE = "mloda.enterprise.licensing.testing.licensed_hmac_binary"

# One distinct marker per hook, so a check that reads the wrong hook is caught too.
_TAMPERED_SIGNATURE_MARKER = "marker-tampered-signature-text"
_MISSING_PLUGINS_CLAIM_MARKER = "marker-missing-plugins-claim-text"
_NOT_YET_VALID_MARKER = "marker-not-yet-valid-license-text"
_UNKNOWN_KID_MARKER = "marker-unknown-kid-license-text"
_TIMESTAMP_WITHOUT_SECONDS_MARKER = "marker-timestamp-without-seconds-license-text"
_COMPACT_TIMESTAMP_MARKER = "marker-compact-timestamp-license-text"
_UNPADDED_MAX_RELEASE_DATE_MARKER = "marker-unpadded-max-release-date-license-text"
_FLOAT_SCHEMA_VERSION_MARKER = "marker-float-schema-version-license-text"


class TestLicenseTokenConformance(LicenseTokenConformanceMixin):
    """Every base check with signed fixtures, plus the token-format checks, against the licensed binary."""

    binary_cmd: ClassVar[list[str]] = [sys.executable, "-m", _LICENSED_BINARY_MODULE]


class _OverriddenTokenVectors(LicenseTokenConformanceMixin):
    """Overrides the eight token hooks with a marker string (not collected: no ``Test`` prefix).
    Each hook is overridden in the form the mixin declares it (property or ``ClassVar``)."""

    @property
    def tampered_signature_text(self) -> str:
        return _TAMPERED_SIGNATURE_MARKER

    missing_plugins_claim_text: ClassVar[str] = _MISSING_PLUGINS_CLAIM_MARKER

    @property
    def timestamp_without_seconds_license_text(self) -> str:
        return _TIMESTAMP_WITHOUT_SECONDS_MARKER

    @property
    def compact_timestamp_license_text(self) -> str:
        return _COMPACT_TIMESTAMP_MARKER

    @property
    def unpadded_max_release_date_license_text(self) -> str:
        return _UNPADDED_MAX_RELEASE_DATE_MARKER

    @property
    def float_schema_version_license_text(self) -> str:
        return _FLOAT_SCHEMA_VERSION_MARKER

    @property
    def not_yet_valid_license_text(self) -> str:
        return _NOT_YET_VALID_MARKER

    @property
    def unknown_kid_license_text(self) -> str:
        return _UNKNOWN_KID_MARKER


def _license_text_that_reached_the_binary(env: dict[str, str], channel: str) -> str:
    """The license text a check handed to the binary via ``channel``: ``MLODA_LICENSE_KEY`` carries it
    inline, ``MLODA_LICENSE_FILE`` names a file holding it. The other channel must be unset."""
    assert {_LICENSE_KEY, _LICENSE_FILE} & env.keys() == {channel}, f"unexpected license channels in env: {env!r}"
    if channel == _LICENSE_FILE:
        return Path(env[_LICENSE_FILE]).read_text(encoding="utf-8")
    return env[_LICENSE_KEY]


@pytest.mark.parametrize(
    ("check_name", "extra_args", "channel", "marker"),
    [
        pytest.param(
            "test_license_token_tampered_is_invalid",
            ("tampered_signature_text",),
            _LICENSE_FILE,
            _TAMPERED_SIGNATURE_MARKER,
            id="tampered_signature",
        ),
        pytest.param(
            "test_license_token_tampered_is_invalid",
            ("missing_plugins_claim_text",),
            _LICENSE_FILE,
            _MISSING_PLUGINS_CLAIM_MARKER,
            id="missing_plugins_claim",
        ),
        pytest.param(
            "test_license_token_tampered_is_invalid",
            ("timestamp_without_seconds_license_text",),
            _LICENSE_FILE,
            _TIMESTAMP_WITHOUT_SECONDS_MARKER,
            id="timestamp_without_seconds",
        ),
        pytest.param(
            "test_license_token_tampered_is_invalid",
            ("compact_timestamp_license_text",),
            _LICENSE_FILE,
            _COMPACT_TIMESTAMP_MARKER,
            id="compact_timestamp",
        ),
        pytest.param(
            "test_license_token_tampered_is_invalid",
            ("unpadded_max_release_date_license_text",),
            _LICENSE_FILE,
            _UNPADDED_MAX_RELEASE_DATE_MARKER,
            id="unpadded_max_release_date",
        ),
        pytest.param(
            "test_license_token_tampered_is_invalid",
            ("float_schema_version_license_text",),
            _LICENSE_FILE,
            _FLOAT_SCHEMA_VERSION_MARKER,
            id="float_schema_version",
        ),
        pytest.param(
            "test_license_not_yet_valid_is_invalid", (), _LICENSE_KEY, _NOT_YET_VALID_MARKER, id="not_yet_valid"
        ),
        pytest.param("test_license_unknown_kid_is_invalid", (), _LICENSE_KEY, _UNKNOWN_KID_MARKER, id="unknown_kid"),
    ],
)
def test_overriding_token_hooks_retargets_the_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    check_name: str,
    extra_args: tuple[str, ...],
    channel: str,
    marker: str,
) -> None:
    """An overridden token hook's text, not the built-in vector, reaches the binary: inline as
    ``MLODA_LICENSE_KEY``, or in the file named by ``MLODA_LICENSE_FILE``, whichever channel the check
    under test uses. ``extra_args`` is the tampered check's hook attribute name."""
    fake = MagicMock(side_effect=RuntimeError("stop-after-capture"))
    monkeypatch.setattr("mloda.testing.binary_model.conformance.run_binary", fake)
    conformance = _OverriddenTokenVectors()
    config_path = tmp_path / "config.json"
    leading_args = (config_path, tmp_path) if channel == _LICENSE_FILE else (config_path,)
    with pytest.raises(RuntimeError, match="stop-after-capture"):
        getattr(conformance, check_name)(*leading_args, *extra_args)
    assert _license_text_that_reached_the_binary(fake.call_args.args[2], channel) == marker


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(valid_license_token(["example_binary"]), False, id="valid"),
        pytest.param(expiring_license_token(["example_binary"], left=timedelta(hours=6)), False, id="expiring"),
        pytest.param(in_grace_license_token(["example_binary"]), True, id="in_grace"),
    ],
)
def test_check_license_token_accepts(text: str, expected: bool) -> None:
    """A plainly valid or soon-expiring token returns False; an in-grace token returns True."""
    assert check_license_token(text, "example_binary") is expected


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(expired_license_token(["example_binary"]), id="expired"),
        pytest.param(valid_license_token(["some_other_plugin"]), id="other_plugin"),
        pytest.param(unknown_kid_license_token(["example_binary"]), id="unknown_kid"),
        pytest.param(simulated_license("example_binary"), id="simulated_license"),
    ],
)
def test_check_license_token_rejects_with_a_reason(text: str) -> None:
    with pytest.raises(FakeLicenseError) as exc_info:
        check_license_token(text, "example_binary")
    assert str(exc_info.value)


@pytest.mark.parametrize("module", [_LICENSED_BINARY_MODULE, _LICENSED_HMAC_BINARY_MODULE])
@pytest.mark.parametrize(
    "flag", [pytest.param("--version", id="version"), pytest.param("--capabilities", id="capabilities")]
)
def test_licensed_binary_light_flags_do_not_import_pyarrow_or_cryptography(module: str, flag: str) -> None:
    """`--version` and `--capabilities` load neither pyarrow nor cryptography; those load only when needed."""
    completed, summary = run_module_probe(module, [flag], ("pyarrow", "cryptography"))
    assert completed.stdout.splitlines()[0]
    assert summary == {"code": 0, "loaded": []}, summary


_HMAC_PLUGIN_ID = "anonymizer_binary"
_HMAC_CMD = [sys.executable, "-m", _LICENSED_HMAC_BINARY_MODULE, "--plugin-id", _HMAC_PLUGIN_ID]
_HMAC_KEY = "42" * 32
_HMAC_ROWS: list[str | None] = ["alice", None, "", "über"]


def _run_licensed_hmac(tmp_path: Path, license_text: str) -> tuple[int, bytes, bytes]:
    config = {
        "input_columns": ["col_a"],
        "operation": "hmac_sha256",
        "parameters": {hmac_sha256_reference.KEY_PARAMETER: _HMAC_KEY},
        "output_columns": {"result": "col_a_token"},
    }
    config_path = write_json(tmp_path / "config.json", config)
    input_bytes = arrow_stream_bytes(pa.schema([pa.field("col_a", pa.string())]), {"col_a": _HMAC_ROWS})
    env = minimal_environment(inherit_license=False, license_key=license_text)
    result = run_binary(_HMAC_CMD, ["run", "--config", str(config_path)], env, input_bytes=input_bytes)
    return result.returncode, result.stdout, result.stderr


def test_licensed_hmac_binary_accepts_a_signed_token_and_matches_the_reference(tmp_path: Path) -> None:
    code, stdout, stderr = _run_licensed_hmac(tmp_path, valid_license_token([_HMAC_PLUGIN_ID]))
    assert code == 0, stderr
    actual = read_arrow_stream(stdout).column("col_a_token").to_pylist()
    assert actual == hmac_sha256_reference.compute_expected_hmac_sha256_column(_HMAC_ROWS, _HMAC_KEY)


def test_licensed_hmac_binary_rejects_the_simulated_license(tmp_path: Path) -> None:
    code, stdout, stderr = _run_licensed_hmac(tmp_path, simulated_license(_HMAC_PLUGIN_ID))
    assert code == LICENSE_INVALID
    error = json.loads(stderr.strip().splitlines()[-1])
    assert error["code"] == LICENSE_INVALID
    assert stdout == b""
