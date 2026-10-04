"""Fixtures for the run manifest tests."""

from __future__ import annotations

import pytest

from mloda.enterprise.extenders.audit.tests import manifest_helpers


@pytest.fixture(params=["hmac", "ed25519"])
def algorithm(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manifest_helpers, "_ALGORITHM", request.param)
