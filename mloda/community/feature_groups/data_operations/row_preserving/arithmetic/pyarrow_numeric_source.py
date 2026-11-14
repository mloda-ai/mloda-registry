"""pyarrow "what counts as numeric" source-column check for arithmetic.

Shared by the point-arithmetic and scalar-arithmetic families so both reject
the same set of non-numeric pyarrow source columns up-front.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa


def pyarrow_non_numeric_descriptor(data: pa.Table, source_col: str) -> Any | None:
    """Return the column type when ``source_col`` is NON-numeric, else ``None``.

    Booleans count as NON-numeric.
    """
    arrow_type = data.column(source_col).type
    if pa.types.is_boolean(arrow_type) or not (pa.types.is_integer(arrow_type) or pa.types.is_floating(arrow_type)):
        return arrow_type
    return None
