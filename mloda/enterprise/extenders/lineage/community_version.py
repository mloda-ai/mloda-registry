"""Version guard: the lineage extender relies on private seams of the community OpenLineage extender."""

from __future__ import annotations

import importlib.metadata

_COMMUNITY = "mloda-community-openlineage"
_ENTERPRISE = "mloda-enterprise"


def mismatch_message() -> str | None:
    """The reason the lineage extender is disabled, or None if the minors match or a version is unknown."""
    try:
        community = importlib.metadata.version(_COMMUNITY)
        enterprise = importlib.metadata.version(_ENTERPRISE)
    except importlib.metadata.PackageNotFoundError:
        return None
    if community.split(".")[:2] == enterprise.split(".")[:2]:
        return None
    return (
        f"{_COMMUNITY} {community} does not match {_ENTERPRISE} {enterprise} (major.minor); "
        "the lineage extender is disabled. Install it with mloda-enterprise[openlineage]."
    )


def require_matching_community() -> None:
    message = mismatch_message()
    if message is not None:
        raise ImportError(message)
