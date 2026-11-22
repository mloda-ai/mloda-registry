"""Conformance checks for a binary that verifies signed license tokens: ``LicenseTokenConformanceMixin``.

Mixed in alongside ``BinaryModelConformanceBase``, it swaps the base's simulated license fixtures for signed
test vectors and adds the token-format checks. Needs ``mloda-testing[binary-model]`` and
``mloda-enterprise[ed25519]``. A host sets its own ``binary_cmd``."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from mloda.enterprise.licensing.testing import license_vectors
from mloda.testing.binary_model.conformance import (
    LICENSE_INVALID,
    BinaryModelConformanceBase,
    assert_error_response,
)

__all__ = ["LicenseTokenConformanceMixin"]


class LicenseTokenConformanceMixin(BinaryModelConformanceBase, suite_base=True):
    """Signed-token fixtures and the token-format checks (spec: Verification), computed lazily from
    ``self.plugin_id``/``self.wrong_plugin_id``. The default fixtures are signed with the published test key
    (``license_vectors.TEST_KID``); a binary built with only production keys must override them."""

    @property
    def valid_license_text(self) -> str:
        return license_vectors.valid_license_token([self.plugin_id])

    @property
    def expired_license_text(self) -> str:
        return license_vectors.expired_license_token([self.plugin_id])

    @property
    def wrong_plugin_license_text(self) -> str:
        return license_vectors.valid_license_token([self.wrong_plugin_id])

    # ``in_grace_license_text`` is time-relative: an override must return a token currently inside its grace window.
    @property
    def in_grace_license_text(self) -> str:
        return license_vectors.in_grace_license_token([self.plugin_id])

    @property
    def tampered_signature_text(self) -> str:
        return license_vectors.tampered_signature_token([self.plugin_id])

    missing_plugins_claim_text: ClassVar[str] = license_vectors.missing_plugins_claim_token()

    @property
    def timestamp_without_seconds_license_text(self) -> str:
        return license_vectors.timestamp_without_seconds_license_token([self.plugin_id])

    @property
    def compact_timestamp_license_text(self) -> str:
        return license_vectors.compact_timestamp_license_token([self.plugin_id])

    @property
    def unpadded_max_release_date_license_text(self) -> str:
        return license_vectors.unpadded_max_release_date_license_token([self.plugin_id])

    @property
    def float_schema_version_license_text(self) -> str:
        return license_vectors.float_schema_version_license_token([self.plugin_id])

    @property
    def not_yet_valid_license_text(self) -> str:
        return license_vectors.not_yet_valid_license_token([self.plugin_id])

    @property
    def unknown_kid_license_text(self) -> str:
        return license_vectors.unknown_kid_license_token([self.plugin_id])

    @pytest.mark.parametrize(
        "attr_name",
        [
            pytest.param("tampered_signature_text", id="tampered_signature"),
            pytest.param("missing_plugins_claim_text", id="missing_plugins_claim"),
            pytest.param("timestamp_without_seconds_license_text", id="timestamp_without_seconds"),
            pytest.param("compact_timestamp_license_text", id="compact_timestamp"),
            pytest.param("unpadded_max_release_date_license_text", id="unpadded_max_release_date"),
            pytest.param("float_schema_version_license_text", id="float_schema_version"),
        ],
    )
    def test_license_token_tampered_is_invalid(self, valid_config_path: Path, tmp_path: Path, attr_name: str) -> None:
        """A broken signature, a well-signed payload missing the required ``plugins`` claim, or a
        well-signed token with a malformed claim format: exit 3 (spec: Verification steps 4, 5; contract:
        License). Parametrized by attribute name, looked up via ``getattr`` at test-run time."""
        self._kit_assert_license_file_rejected(valid_config_path, tmp_path, getattr(self, attr_name))

    def test_license_not_yet_valid_is_invalid(self, valid_config_path: Path) -> None:
        """A token whose ``nbf`` lies in the future: exit 3, not yet valid (spec: Verification
        step 6; contract: License)."""
        env = self.platform_env({"MLODA_LICENSE_KEY": self.not_yet_valid_license_text})
        result = self._kit_run_with_config(valid_config_path, env)
        assert_error_response(result, LICENSE_INVALID)

    def test_license_unknown_kid_is_invalid(self, valid_config_path: Path) -> None:
        """A well-signed token under a ``kid`` the verifier's key map does not contain: exit 3
        (spec: Verification step 3; contract: License)."""
        env = self.platform_env({"MLODA_LICENSE_KEY": self.unknown_kid_license_text})
        result = self._kit_run_with_config(valid_config_path, env)
        assert_error_response(result, LICENSE_INVALID)
