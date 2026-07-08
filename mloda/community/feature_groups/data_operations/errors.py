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


def _build_unsupported_value_error(
    value: str,
    supported: Iterable[str],
    *,
    value_label: str,
    supported_plural: str,
    framework: str | None = None,
    operation: str | None = None,
) -> ValueError:
    """Build a ``ValueError`` for an unsupported value of a given label.

    Internal helper. The public ``unsupported_*_error`` functions
    delegate to this; they exist as thin wrappers that fix the
    ``value_label`` and ``supported_plural`` strings.
    """
    prefix = f"Unsupported {value_label}"
    if framework is not None:
        prefix += f" for {framework}"
    if operation is not None:
        prefix += f" {operation}"
    supported_list = ", ".join(sorted(set(supported)))
    return ValueError(f"{prefix}: {value!r}. Supported {supported_plural}: {supported_list}.")


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
    return _build_unsupported_value_error(
        agg_type,
        supported,
        value_label="aggregation type",
        supported_plural="types",
        framework=framework,
        operation=operation,
    )


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
    return _build_unsupported_value_error(
        frame_type,
        supported,
        value_label="frame type",
        supported_plural="types",
        framework=framework,
    )


def unsupported_op_error(
    op: str,
    supported: Iterable[str],
    *,
    framework: str | None = None,
) -> ValueError:
    """Build a ``ValueError`` describing an unsupported operation.

    Args:
        op: The value the caller provided.
        supported: All ``op`` values the caller *could* have used.
            Deduplicated and sorted alphabetically in the message.
        framework: Optional framework label, included in the message when
            provided.
    """
    return _build_unsupported_value_error(
        op,
        supported,
        value_label="operation",
        supported_plural="operations",
        framework=framework,
    )


def unsupported_subtype_error(
    subtype: str,
    supported: Iterable[str],
    *,
    operation: str,
) -> ValueError:
    """Build a ``ValueError`` describing an unsupported subtype of a data operation.

    Args:
        subtype: The value the caller provided.
        supported: All subtype values the caller *could* have used.
            Deduplicated and sorted alphabetically in the message.
        operation: The operation name (``"aggregation"``, ``"rank"``, ...),
            included in the value label so the user knows which operation
            rejected the subtype.
    """
    return _build_unsupported_value_error(
        subtype,
        supported,
        value_label=f"{operation} subtype",
        supported_plural="subtypes",
    )
