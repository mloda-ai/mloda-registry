"""polars "what counts as numeric" source-column check for arithmetic.

Shared by the point-arithmetic and scalar-arithmetic families so both reject
the same set of non-numeric polars source columns up-front.
"""

from __future__ import annotations

from typing import Any

import polars as pl


def polars_non_numeric_descriptor(dtype: Any) -> Any | None:
    """Return ``dtype`` when the polars dtype is NON-numeric, else ``None``.

    Booleans count as NON-numeric.
    """
    if dtype == pl.Boolean or not dtype.is_numeric():
        return dtype
    return None
