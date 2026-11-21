"""Reads back the license text a conformance check handed to a binary, shared by the retarget tests."""

from __future__ import annotations

from pathlib import Path

LICENSE_KEY = "MLODA_LICENSE_KEY"
LICENSE_FILE = "MLODA_LICENSE_FILE"


def license_text_that_reached_the_binary(env: dict[str, str], channel: str) -> str:
    """The license text a check handed to the binary via ``channel``: ``MLODA_LICENSE_KEY`` carries it
    inline, ``MLODA_LICENSE_FILE`` names a file holding it. The other channel must be unset."""
    assert {LICENSE_KEY, LICENSE_FILE} & env.keys() == {channel}, f"unexpected license channels in env: {env!r}"
    if channel == LICENSE_FILE:
        return Path(env[LICENSE_FILE]).read_text(encoding="utf-8")
    return env[LICENSE_KEY]
