"""pandas "what counts as numeric" source-column check for arithmetic.

Shared by the point-arithmetic and scalar-arithmetic families so both reject
the same set of non-numeric pandas source columns up-front.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def pandas_non_numeric_descriptor(series: Any) -> Any | None:
    """Return ``series.dtype`` when the pandas series is NON-numeric, else ``None``.

    Booleans count as NON-numeric.
    """
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        return series.dtype
    return None
