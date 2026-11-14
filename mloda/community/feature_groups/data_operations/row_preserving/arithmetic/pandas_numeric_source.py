"""pandas "what counts as numeric" source-column check for arithmetic.

Shared by the point-arithmetic and scalar-arithmetic families so both reject
the same set of non-numeric pandas source columns up-front.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def pandas_non_numeric_descriptor(data: pd.DataFrame, source_col: str) -> Any | None:
    """Return the column dtype when ``source_col`` is NON-numeric, else ``None``.

    Booleans count as NON-numeric.
    """
    series = data[source_col]
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        return series.dtype
    return None
