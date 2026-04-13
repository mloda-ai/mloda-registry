"""Helpers that build uniform, user-guiding errors for data operations.

Every ``_compute_*`` implementation branches on ``agg_type`` (and for
frame aggregates on ``frame_type``). When a user passes a value that the
concrete framework does not support, a bare ``ValueError`` that echoes
the rejected value is unhelpful: the user has no way to discover which
values *are* accepted without reading the source.

The helpers in this module produce messages that include the rejected
value and the sorted list of supported values for the specific framework
and operation that raised. The messages are deterministic so that tests
can match them reliably.
"""

from __future__ import annotations

from collections.abc import Iterable


def unsupported_agg_type_error(
    agg_type: str,
    supported: Iterable[str],
    *,
    framework: str | None = None,
    operation: str | None = None,
) -> ValueError:
    """Build a ``ValueError`` describing an unsupported aggregation type.

    Args:
        agg_type: The value the caller provided.
        supported: All ``agg_type`` values the caller *could* have used.
            Deduplicated and sorted alphabetically in the message.
        framework: Optional framework label (``"DuckDB"``, ``"SQLite"``,
            ``"Pandas"``, ...). Included in the message when provided so
            that the user knows which backend rejected the value.
        operation: Optional operation qualifier (``"frame aggregate"``,
            ``"cumulative/expanding"``, ...) that disambiguates frameworks
            that implement more than one operation.
    """
    prefix = "Unsupported aggregation type"
    if framework is not None:
        prefix += f" for {framework}"
    if operation is not None:
        prefix += f" {operation}"
    supported_list = ", ".join(sorted(set(supported)))
    return ValueError(f"{prefix}: {agg_type!r}. Supported types: {supported_list}.")


def unsupported_frame_type_error(
    frame_type: str,
    supported: Iterable[str],
    *,
    framework: str | None = None,
) -> ValueError:
    """Build a ``ValueError`` describing an unsupported frame type.

    Args:
        frame_type: The value the caller provided.
        supported: All ``frame_type`` values the caller *could* have used.
            Deduplicated and sorted alphabetically in the message.
        framework: Optional framework label, included in the message when
            provided.
    """
    prefix = "Unsupported frame type"
    if framework is not None:
        prefix += f" for {framework}"
    supported_list = ", ".join(sorted(set(supported)))
    return ValueError(f"{prefix}: {frame_type!r}. Supported types: {supported_list}.")
