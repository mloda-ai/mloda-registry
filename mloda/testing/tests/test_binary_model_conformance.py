"""Wires the class-based binary-model conformance kit to our own simulated CLI stub.

Every class attribute default in ``BinaryModelConformanceBase``/``HashOperationConformanceMixin``
already points at ``mloda.testing.binary_model.simulated_binary``, so nothing needs overriding
here; a future conformance run against a real binary reuses these classes unmodified by
subclassing with a different ``binary_cmd``. The file also checks that the kit's license-state
hooks are overridable."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mloda.testing.binary_model.conformance import BinaryModelConformanceBase, HashOperationConformanceMixin

_IN_GRACE_MARKER = "marker-in-grace-license-text"
_NOT_YET_VALID_MARKER = "marker-not-yet-valid-license-text"
_UNKNOWN_KID_MARKER = "marker-unknown-kid-license-text"


class TestBinaryModelConformance(HashOperationConformanceMixin, BinaryModelConformanceBase):
    pass


class _OverriddenLicenseVectors(BinaryModelConformanceBase):
    """Overrides the license-state hooks with marker strings (not collected: no ``Test`` prefix)."""

    @property
    def in_grace_license_text(self) -> str:
        return _IN_GRACE_MARKER

    @property
    def not_yet_valid_license_text(self) -> str:
        return _NOT_YET_VALID_MARKER

    @property
    def unknown_kid_license_text(self) -> str:
        return _UNKNOWN_KID_MARKER


@pytest.mark.parametrize(
    ("check_name", "marker"),
    [
        pytest.param("test_license_in_grace_is_accepted", _IN_GRACE_MARKER, id="in_grace"),
        pytest.param("test_license_not_yet_valid_is_invalid", _NOT_YET_VALID_MARKER, id="not_yet_valid"),
        pytest.param("test_license_unknown_kid_is_invalid", _UNKNOWN_KID_MARKER, id="unknown_kid"),
    ],
)
def test_overriding_license_vectors_retargets_the_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, check_name: str, marker: str
) -> None:
    """An overridden license-state hook's token, not the built-in vector, reaches the binary as
    ``MLODA_LICENSE_KEY`` (contract: License)."""
    fake = MagicMock(side_effect=RuntimeError("stop-after-capture"))
    monkeypatch.setattr("mloda.testing.binary_model.conformance.run_binary", fake)
    conformance = _OverriddenLicenseVectors()
    with pytest.raises(RuntimeError, match="stop-after-capture"):
        getattr(conformance, check_name)(tmp_path / "config.json")
    assert fake.call_args.args[2]["MLODA_LICENSE_KEY"] == marker


def test_size_cap_constants_are_exported() -> None:
    """`MESSAGE_MAX_BYTES` and `STDERR_SOFT_CAP_BYTES` belong on the conformance kit's public
    surface, re-exported via `__all__` for external consumers (contract: Data handling)."""
    from mloda.testing.binary_model import conformance

    assert "MESSAGE_MAX_BYTES" in conformance.__all__
    assert "STDERR_SOFT_CAP_BYTES" in conformance.__all__
    assert "COLUMN_TYPES" in conformance.__all__
    assert [name for name in conformance.__all__ if not hasattr(conformance, name)] == []
