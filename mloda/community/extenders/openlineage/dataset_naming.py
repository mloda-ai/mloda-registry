"""Maps a load's data access identity to an OpenLineage dataset (namespace, name) per the naming spec."""

from __future__ import annotations

import os

from openlineage.client.naming.dataset import ABFSS, GCS, S3, DatasetNaming, LocalFileSystem

_ABFSS_SUFFIX = ".dfs.core.windows.net"


def load_dataset(identity: str, fallback_namespace: str) -> tuple[str, str]:
    """(namespace, name) for s3, gs, abfss, file URIs and existing absolute paths; anything else, or any
    failure, is (fallback_namespace, identity). Never raises."""
    try:
        naming = _naming(identity)
    except Exception:
        naming = None
    if naming is None:
        return fallback_namespace, identity
    try:
        return naming.get_namespace(), naming.get_name()
    except Exception:
        return fallback_namespace, identity


def _naming(identity: str) -> DatasetNaming | None:
    if "?" in identity or "#" in identity:
        return None
    scheme, sep, rest = identity.partition("://")
    if not sep:
        if identity.startswith("/") and os.path.exists(identity):
            return LocalFileSystem(path=identity) if "@" not in identity else None
        return None
    scheme = scheme.lower()
    authority, slash, path = rest.partition("/")
    path = path if slash else ""
    if scheme == "abfss":
        container, at, host = authority.partition("@")
        if not at or "@" in host or ":" in host or not host.endswith(_ABFSS_SUFFIX):
            return None
        return ABFSS(container=container, service=host[: -len(_ABFSS_SUFFIX)], path=path)
    if "@" in authority or "@" in path:
        return None
    if scheme == "file":
        return LocalFileSystem(path=f"/{path}") if not authority and slash else None
    if ":" in authority:
        return None
    if scheme == "s3":
        return S3(bucket_name=authority, object_key=path)
    if scheme == "gs":
        return GCS(bucket_name=authority, object_key=path)
    return None
