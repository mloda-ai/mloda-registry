"""Runs ``LicenseTokenConformanceMixin`` against the licensed binary in
``mloda.enterprise.licensing.testing.licensed_binary``, plus the token hooks' retargeting, the in-process
``check_license_token`` and the light flags of the licensed binary.
Needs ``mloda-testing[binary-model]`` and ``mloda-enterprise[ed25519]``."""

from __future__ import annotations

import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import ClassVar

import pyarrow as pa
import pytest

from mloda.community.feature_groups.binary_model import binary
from mloda.community.feature_groups.binary_model.mixin import BinaryModelMixin
from mloda.enterprise.licensing.testing.conformance import LicenseTokenConformanceMixin
from mloda.enterprise.licensing.testing.license_vectors import (
    expired_license_token,
    expiring_license_token,
    in_grace_license_token,
    unknown_kid_license_token,
    valid_license_token,
)
from mloda.enterprise.licensing.testing.licensed_binary import check_license_token
from mloda.testing.binary_model.conformance import (
    BinaryModelConformanceBase,
    HashOperationConformanceMixin,
)
from mloda.testing.binary_model.simulated_binary import FakeLicenseError, simulated_license
from mloda.testing.tests._license_channel import LICENSE_FILE, LICENSE_KEY, license_text_a_check_hands_to_the_binary
from mloda.testing.tests._module_probe import run_module_probe

_LICENSED_BINARY_MODULE = "mloda.enterprise.licensing.testing.licensed_binary"

# One distinct marker per hook, so a check that reads the wrong hook is caught too.
_TAMPERED_SIGNATURE_MARKER = "marker-tampered-signature-text"
_MISSING_PLUGINS_CLAIM_MARKER = "marker-missing-plugins-claim-text"
_NOT_YET_VALID_MARKER = "marker-not-yet-valid-license-text"
_UNKNOWN_KID_MARKER = "marker-unknown-kid-license-text"
_TIMESTAMP_WITHOUT_SECONDS_MARKER = "marker-timestamp-without-seconds-license-text"
_COMPACT_TIMESTAMP_MARKER = "marker-compact-timestamp-license-text"
_UNPADDED_MAX_RELEASE_DATE_MARKER = "marker-unpadded-max-release-date-license-text"
_FLOAT_SCHEMA_VERSION_MARKER = "marker-float-schema-version-license-text"


class TestLicenseTokenConformance(
    HashOperationConformanceMixin, LicenseTokenConformanceMixin, BinaryModelConformanceBase
):
    """The documented composition for a token-verifying hash binary: the hash checks, the token-format checks
    and every base check with signed fixtures, against the licensed binary."""

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


@pytest.mark.parametrize(
    ("check_name", "extra_args", "channel", "marker"),
    [
        pytest.param(
            "test_license_token_tampered_is_invalid",
            ("tampered_signature_text",),
            LICENSE_FILE,
            _TAMPERED_SIGNATURE_MARKER,
            id="tampered_signature",
        ),
        pytest.param(
            "test_license_token_tampered_is_invalid",
            ("missing_plugins_claim_text",),
            LICENSE_FILE,
            _MISSING_PLUGINS_CLAIM_MARKER,
            id="missing_plugins_claim",
        ),
        pytest.param(
            "test_license_token_tampered_is_invalid",
            ("timestamp_without_seconds_license_text",),
            LICENSE_FILE,
            _TIMESTAMP_WITHOUT_SECONDS_MARKER,
            id="timestamp_without_seconds",
        ),
        pytest.param(
            "test_license_token_tampered_is_invalid",
            ("compact_timestamp_license_text",),
            LICENSE_FILE,
            _COMPACT_TIMESTAMP_MARKER,
            id="compact_timestamp",
        ),
        pytest.param(
            "test_license_token_tampered_is_invalid",
            ("unpadded_max_release_date_license_text",),
            LICENSE_FILE,
            _UNPADDED_MAX_RELEASE_DATE_MARKER,
            id="unpadded_max_release_date",
        ),
        pytest.param(
            "test_license_token_tampered_is_invalid",
            ("float_schema_version_license_text",),
            LICENSE_FILE,
            _FLOAT_SCHEMA_VERSION_MARKER,
            id="float_schema_version",
        ),
        pytest.param(
            "test_license_not_yet_valid_is_invalid", (), LICENSE_KEY, _NOT_YET_VALID_MARKER, id="not_yet_valid"
        ),
        pytest.param("test_license_unknown_kid_is_invalid", (), LICENSE_KEY, _UNKNOWN_KID_MARKER, id="unknown_kid"),
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
    assert (
        license_text_a_check_hands_to_the_binary(
            monkeypatch, tmp_path, _OverriddenTokenVectors(), check_name, extra_args, channel
        )
        == marker
    )


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
    ("text", "reason_prefix"),
    [
        pytest.param(expired_license_token(["example_binary"]), "license expired: exp was ", id="expired"),
        pytest.param(
            valid_license_token(["some_other_plugin"]),
            "license does not entitle plugin 'example_binary'",
            id="other_plugin",
        ),
        pytest.param(
            unknown_kid_license_token(["example_binary"]),
            "unknown license key id (kid) 'test-unknown'",
            id="unknown_kid",
        ),
        pytest.param(
            simulated_license("example_binary"), "license token is not a PASETO v4.public token", id="simulated_license"
        ),
    ],
)
def test_check_license_token_rejects_with_a_reason(text: str, reason_prefix: str) -> None:
    """Each rejection carries the verifier's own reason, never the token text."""
    with pytest.raises(FakeLicenseError) as exc_info:
        check_license_token(text, "example_binary")
    reason = str(exc_info.value)
    assert reason.startswith(reason_prefix), reason
    assert text not in reason


@pytest.mark.parametrize(
    "flag", [pytest.param("--version", id="version"), pytest.param("--capabilities", id="capabilities")]
)
def test_licensed_binary_light_flags_do_not_import_pyarrow_numpy_or_cryptography(flag: str) -> None:
    """`--version` and `--capabilities` load none of pyarrow, numpy and cryptography; those load only when needed."""
    completed, summary = run_module_probe(_LICENSED_BINARY_MODULE, [flag], ("pyarrow", "numpy", "cryptography"))
    assert completed.stdout.splitlines()[0]
    assert summary == {"code": 0, "loaded": []}, summary


@pytest.fixture(autouse=True)
def _clear_capability_cache_before_each_test() -> None:
    binary.clear_capability_cache()


class _LicensedStubModel(BinaryModelMixin):
    BINARY_PLUGIN_ID = "example_binary"
    BINARY_COMMAND_OVERRIDE = [sys.executable, "-m", _LICENSED_BINARY_MODULE]
    LICENSE_KEY_OVERRIDE = expiring_license_token(["example_binary"], left=timedelta(hours=6))


def test_mixin_run_with_a_signed_token_logs_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    """A soon-expiring but still valid signed token runs through the mixin silently, no grace warning."""
    with caplog.at_level(logging.DEBUG, logger="mloda.community.feature_groups.binary_model"):
        result = _LicensedStubModel.run_binary_model(
            pa.table({"col_a": ["alpha"]}), ["col_a"], "hash", {}, {"result": "col_a_hash"}
        )
    assert result.column("col_a_hash").to_pylist()
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
