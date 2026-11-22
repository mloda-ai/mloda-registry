"""Reads back the license text a conformance check handed to a binary, shared by the retarget tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

LICENSE_KEY = "MLODA_LICENSE_KEY"
LICENSE_FILE = "MLODA_LICENSE_FILE"


def license_text_that_reached_the_binary(env: dict[str, str], channel: str) -> str:
    """The license text a check handed to the binary via ``channel``: ``MLODA_LICENSE_KEY`` carries it
    inline, ``MLODA_LICENSE_FILE`` names a file holding it. The other channel must be unset."""
    assert {LICENSE_KEY, LICENSE_FILE} & env.keys() == {channel}, f"unexpected license channels in env: {env!r}"
    if channel == LICENSE_FILE:
        return Path(env[LICENSE_FILE]).read_text(encoding="utf-8")
    return env[LICENSE_KEY]


def license_text_a_check_hands_to_the_binary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    conformance: Any,
    check_name: str,
    extra_args: tuple[str, ...],
    channel: str,
) -> str:
    """Run `check_name` with `run_binary` stubbed and return the license text it handed over."""
    fake = MagicMock(side_effect=RuntimeError("stop-after-capture"))
    monkeypatch.setattr("mloda.testing.binary_model.conformance.run_binary", fake)
    config_path = tmp_path / "config.json"
    leading_args = (config_path, tmp_path) if channel == LICENSE_FILE else (config_path,)
    with pytest.raises(RuntimeError, match="stop-after-capture"):
        getattr(conformance, check_name)(*leading_args, *extra_args)
    return license_text_that_reached_the_binary(fake.call_args.args[2], channel)
