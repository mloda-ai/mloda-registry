"""Sanitizing of data-access identities before extenders record or publish them."""

from __future__ import annotations

import re
from typing import Any

_URI_SCHEME = re.compile(r"[A-Za-z][\w+.:-]*")
_QUERY_OR_FRAGMENT = re.compile(r"[?#]")
_PATH_PARAMS = re.compile(r"[;&]")
_AUTHORITY_LEAK = re.compile(r"[;&=\s]")


def sanitize_data_access_identity(identity: str) -> str:
    # The authority and path are cut at leak markers too. Userinfo goes first: ; and & are valid inside it.
    scheme, separator, rest = identity.partition("://")
    if not separator or not _URI_SCHEME.fullmatch(scheme):
        return identity
    cut = _QUERY_OR_FRAGMENT.split(rest, maxsplit=1)[0]
    if rest.partition("/")[0].rfind("@") >= len(cut):  # a ? or # sits inside the userinfo, so strip it first
        rest = _QUERY_OR_FRAGMENT.split(rest.rpartition("@")[2], maxsplit=1)[0]
    else:
        rest = cut.rpartition("@")[2]
    authority, slash, path = rest.partition("/")
    path = _PATH_PARAMS.split(path, maxsplit=1)[0]
    cut_authority = _AUTHORITY_LEAK.split(authority, maxsplit=1)[0]
    if cut_authority != authority:
        return f"{scheme}://{cut_authority}"
    return f"{scheme}://{authority}{slash}{path}"


def resolve_data_access_identity(args: tuple[Any, ...], context_identity: str | None) -> str | None:
    if context_identity is None:  # the context identity is the recording gate, even if args[0] is a str
        return None
    # Core passes the raw data_access first; using the raw string keeps what core's default-deny
    # identity reduces to a type name.
    raw = args[0] if args and isinstance(args[0], str) else context_identity
    sanitized = sanitize_data_access_identity(raw)
    if sanitized == raw and raw != context_identity and "://" in raw:
        # raw is scheme-shaped but malformed, so it went unsanitized; fall back to core's identity.
        return sanitize_data_access_identity(context_identity)
    return sanitized
