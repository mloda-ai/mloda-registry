"""Wires the class-based binary-model conformance kit to our own simulated CLI stub.

Every class attribute default in ``BinaryModelConformanceBase``/``HashOperationConformanceMixin``
already points at ``mloda.testing.binary_model.simulated_binary``, so nothing needs overriding
here; a future conformance run against a real binary reuses these classes unmodified by
subclassing with a different ``binary_cmd``. The file also checks that the kit's license-state
hooks are overridable."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from mloda.testing.binary_model.conformance import (
    DATA_ERROR,
    DATA_FREE_MARKER,
    USAGE_ERROR,
    BinaryModelConformanceBase,
    HashOperationConformanceMixin,
    arrow_stream_bytes,
    arrow_stream_bytes_invalid_utf8,
    assert_error_response,
    read_arrow_stream,
    run_binary,
    stderr_error_object,
    write_text,
)

_LICENSE_KEY = "MLODA_LICENSE_KEY"
_LICENSE_FILE = "MLODA_LICENSE_FILE"

# One distinct marker per hook, so a check that reads the wrong hook is caught too.
_VALID_MARKER = "marker-valid-license-text"
_EXPIRED_MARKER = "marker-expired-license-text"
_WRONG_PLUGIN_MARKER = "marker-wrong-plugin-license-text"
_UNPARSEABLE_MARKER = "marker-tampered-unparseable-text"
_TAMPERED_SIGNATURE_MARKER = "marker-tampered-signature-text"
_MISSING_PLUGINS_CLAIM_MARKER = "marker-missing-plugins-claim-text"
_IN_GRACE_MARKER = "marker-in-grace-license-text"
_NOT_YET_VALID_MARKER = "marker-not-yet-valid-license-text"
_UNKNOWN_KID_MARKER = "marker-unknown-kid-license-text"


class TestBinaryModelConformance(HashOperationConformanceMixin, BinaryModelConformanceBase):
    def test_simulated_binary_input_invalid_utf8_is_data_error(
        self, valid_config_path: Path, valid_license_env: dict[str, str]
    ) -> None:
        """A utf8 input value backed by invalid bytes is malformed data (exit 5), not an internal
        error (exit 6). Simulated-binary regression only, so it lives here and not in the kit base."""
        result = run_binary(
            self.binary_cmd,
            ["run", "--config", str(valid_config_path)],
            valid_license_env,
            input_bytes=arrow_stream_bytes_invalid_utf8(
                self.default_input_columns[0], DATA_FREE_MARKER.encode("utf-8") + b"\xff"
            ),
            timeout=self.binary_timeout_seconds,
        )
        assert_error_response(result, DATA_ERROR)
        assert DATA_FREE_MARKER.encode("utf-8") not in result.stderr

    @pytest.mark.parametrize(
        "config_text",
        [
            pytest.param("9" * 5000, id="oversized_int"),
            pytest.param("[" * 100000, id="deeply_nested"),
        ],
    )
    def test_simulated_binary_config_json_module_cannot_parse_is_usage_error(
        self, valid_license_env: dict[str, str], tmp_path: Path, config_text: str
    ) -> None:
        """A `--config` file whose whole content is text ``json.loads`` itself cannot parse without
        raising -- an oversized integer literal, or JSON deep enough to raise ``RecursionError`` --
        is a usage error (exit 1), the same as any other malformed config, not an uncaught exception
        reported as an internal error. Simulated-binary regression only, so it lives here and not in
        the kit base (contract: Configuration, Errors)."""
        config_path = write_text(tmp_path / "config.json", config_text)
        input_bytes = arrow_stream_bytes(self.default_input_schema(), self.default_input_rows())
        result = run_binary(
            self.binary_cmd,
            ["run", "--config", str(config_path)],
            valid_license_env,
            input_bytes=input_bytes,
            timeout=self.binary_timeout_seconds,
        )
        assert_error_response(result, USAGE_ERROR)


class _OverriddenLicenseVectors(BinaryModelConformanceBase):
    """Overrides every license-state hook with a marker string (not collected: no ``Test`` prefix).
    Each hook is overridden in the form the base declares it (property or ``ClassVar``)."""

    @property
    def valid_license_text(self) -> str:
        return _VALID_MARKER

    @property
    def expired_license_text(self) -> str:
        return _EXPIRED_MARKER

    @property
    def wrong_plugin_license_text(self) -> str:
        return _WRONG_PLUGIN_MARKER

    tampered_unparseable_text: ClassVar[str] = _UNPARSEABLE_MARKER

    @property
    def tampered_signature_text(self) -> str:
        return _TAMPERED_SIGNATURE_MARKER

    missing_plugins_claim_text: ClassVar[str] = _MISSING_PLUGINS_CLAIM_MARKER

    @property
    def in_grace_license_text(self) -> str:
        return _IN_GRACE_MARKER

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
        pytest.param("test_license_accepted_via_license_key_inline", (), _LICENSE_KEY, _VALID_MARKER, id="valid"),
        pytest.param("test_license_expired_is_invalid", (), _LICENSE_FILE, _EXPIRED_MARKER, id="expired"),
        pytest.param("test_license_wrong_plugin_is_invalid", (), _LICENSE_KEY, _WRONG_PLUGIN_MARKER, id="wrong_plugin"),
        pytest.param(
            "test_license_tampered_is_invalid",
            ("tampered_unparseable_text",),
            _LICENSE_FILE,
            _UNPARSEABLE_MARKER,
            id="unparseable_text",
        ),
        pytest.param(
            "test_license_tampered_is_invalid",
            ("tampered_signature_text",),
            _LICENSE_FILE,
            _TAMPERED_SIGNATURE_MARKER,
            id="tampered_signature",
        ),
        pytest.param(
            "test_license_tampered_is_invalid",
            ("missing_plugins_claim_text",),
            _LICENSE_FILE,
            _MISSING_PLUGINS_CLAIM_MARKER,
            id="missing_plugins_claim",
        ),
        pytest.param("test_license_in_grace_is_accepted", (), _LICENSE_KEY, _IN_GRACE_MARKER, id="in_grace"),
        pytest.param(
            "test_license_not_yet_valid_is_invalid", (), _LICENSE_KEY, _NOT_YET_VALID_MARKER, id="not_yet_valid"
        ),
        pytest.param("test_license_unknown_kid_is_invalid", (), _LICENSE_KEY, _UNKNOWN_KID_MARKER, id="unknown_kid"),
    ],
)
def test_overriding_license_vectors_retargets_the_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    check_name: str,
    extra_args: tuple[str, ...],
    channel: str,
    marker: str,
) -> None:
    """An overridden license-state hook's token, not the built-in vector, reaches the binary: inline
    as ``MLODA_LICENSE_KEY``, or in the file named by ``MLODA_LICENSE_FILE``, whichever channel the
    check under test uses (contract: License). Checks delivering via a file write it under
    ``tmp_path`` and so also take it; ``extra_args`` are any trailing arguments (the tampered check's
    hook attribute name)."""
    fake = MagicMock(side_effect=RuntimeError("stop-after-capture"))
    monkeypatch.setattr("mloda.testing.binary_model.conformance.run_binary", fake)
    conformance = _OverriddenLicenseVectors()
    config_path = tmp_path / "config.json"
    leading_args = (config_path, tmp_path) if channel == _LICENSE_FILE else (config_path,)
    with pytest.raises(RuntimeError, match="stop-after-capture"):
        getattr(conformance, check_name)(*leading_args, *extra_args)
    assert _license_text_that_reached_the_binary(fake.call_args.args[2], channel) == marker


@pytest.mark.parametrize(
    "last_line",
    [
        pytest.param(b"9" * 5000, id="oversized_int"),
        pytest.param(b"[" * 100000, id="deeply_nested"),
        pytest.param(b"oops", id="not_json"),
    ],
)
def test_stderr_error_object_unparseable_last_line_fails_an_assertion(last_line: bytes) -> None:
    """``stderr_error_object`` must fail with an ``AssertionError`` on a last stderr line
    ``json.loads`` cannot parse -- an oversized integer, JSON deep enough to raise
    ``RecursionError``, or plain garbage -- not let ``ValueError``/``RecursionError``/
    ``json.JSONDecodeError`` escape (contract: Errors)."""
    with pytest.raises(AssertionError):
        stderr_error_object(last_line + b"\n")


def test_stderr_error_object_message_with_unicode_line_boundary_returns_full_message() -> None:
    """The last stderr line is found by splitting on ``\\n`` only, not on ``str.splitlines()``'s
    wider notion of a line boundary (U+2028, U+2029, ...): a message containing a raw U+2028
    must come back whole (contract: Errors, Data handling)."""
    message = "bad column a b"
    stderr = json.dumps({"code": 5, "message": message}, ensure_ascii=False).encode("utf-8") + b"\n"
    error = stderr_error_object(stderr)
    assert error["message"] == message


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"garbage", id="not_arrow_at_all"),
        pytest.param(None, id="invalid_utf8_value"),
    ],
)
def test_read_arrow_stream_malformed_input_fails_an_assertion(data: bytes | None) -> None:
    """``read_arrow_stream`` must fully validate its input and fail with an ``AssertionError``,
    not let an Arrow-level error escape or silently return a table over invalid data (contract:
    Data)."""
    payload = data if data is not None else arrow_stream_bytes_invalid_utf8()
    with pytest.raises(AssertionError):
        read_arrow_stream(payload)


def test_size_cap_constants_are_exported() -> None:
    """`MESSAGE_MAX_BYTES` and `STDERR_SOFT_CAP_BYTES` belong on the conformance kit's public
    surface, re-exported via `__all__` for external consumers (contract: Data handling)."""
    from mloda.testing.binary_model import conformance

    assert "MESSAGE_MAX_BYTES" in conformance.__all__
    assert "STDERR_SOFT_CAP_BYTES" in conformance.__all__
    assert "COLUMN_TYPES" in conformance.__all__
    assert [name for name in conformance.__all__ if not hasattr(conformance, name)] == []
