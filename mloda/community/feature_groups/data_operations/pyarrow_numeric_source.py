"""pyarrow "what counts as numeric" source-column check for arithmetic.

Shared by the point-arithmetic and scalar-arithmetic families so both reject
the same set of non-numeric pyarrow source columns up-front.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa


def pyarrow_non_numeric_descriptor(arrow_type: Any) -> Any | None:
    """Return ``arrow_type`` when the pyarrow type is NON-numeric, else ``None``.

    Booleans count as NON-numeric.
    """
    if pa.types.is_boolean(arrow_type) or not (pa.types.is_integer(arrow_type) or pa.types.is_floating(arrow_type)):
        return arrow_type
    return None
