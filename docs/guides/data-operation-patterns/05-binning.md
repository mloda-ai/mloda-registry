# Binning

Binning maps a numeric column onto integer bucket indices `0..n-1`. Two variants ship: equal-width (`bin`) and quantile-based (`qbin`).

**What**: `BinningFeatureGroup` (an `OperationFeatureGroup` whose loop is `assign_bins`) accepts feature names of the form `{col}__bin_{N}` or `{col}__qbin_{N}`; each backend (`PandasBinning`, `DuckdbBinning`, ...) is an `Implementation` of it.
**When**: You need discretization for downstream grouping, charting, or model features.
**Why**: Both operations are standard, but implementing them identically across frameworks is non-trivial because of NULL propagation.
**Where**: `mloda/community/feature_groups/data_operations/row_preserving/binning/`.
**How**: Feature name encodes the op and bin count; the framework implementation computes the bin index per row and preserves input order.

---

## Two binning modes

| Mode | Semantics | Result shape |
|---|---|---|
| `bin` (equal-width) | Divide `[min, max]` into N intervals of equal width. Bin index = `floor((value - min) / width)`, clamped to `N-1`. | Bins may be unbalanced if the distribution is skewed. |
| `qbin` (quantile) | Sort rows by value, assign rank `r` to bin `r * N // n` where `n` is the count of non-null values. | Bins are always roughly equal in row count. |

`qbin` uses rank-based assignment rather than sample quantiles. That is a deliberate choice to sidestep interpolation disagreements across frameworks: ranks are integers and leave no room for numerical drift.

---

## Source column

A non-numeric or boolean source raises `ValueError` (SQLite booleans excepted, see [known divergences](known-divergences.md)). `bin` also raises for an integer beyond 2**53, which float64 would round; `qbin` accepts it.

---

## SQL qbin is not NTILE

`NTILE(N)` is not equivalent to `r * N // n`: it spreads the remainder over the first buckets (6 rows into 4 bins gives `[0,0,1,1,2,3]`, the formula gives `[0,0,1,2,2,3]`) and leaves no gaps when there are more bins than rows. The SQL backends therefore compute `(ROW_NUMBER() - 1) * N // COUNT` over the non-null rows instead, so every backend resolves to the same labels. `bin` computes the width and offsets in float64 on every backend.

---

## NULL and NaN handling

- Input NULLs are skipped when computing `min`, `max`, and the non-null count `n`.
- Rows with NULL in the source column receive NULL in the bin column, not a real bin index.
- NaN in floating columns is treated like NULL. DuckDB explicitly guards with `isnan(col)`:

```sql
CASE WHEN col IS NULL OR isnan(col) THEN NULL
     ELSE (ROW_NUMBER() OVER (
         PARTITION BY CASE WHEN col IS NOT NULL AND NOT isnan(col) THEN 1 END
         ORDER BY col, input_position) - 1) * N
          // COUNT(col) FILTER (WHERE NOT isnan(col)) OVER () END
```

The `PARTITION BY CASE WHEN ...` clause is what ensures NULL/NaN rows do not participate in the rank at all; they stay unpartitioned and the CASE returns NULL for them.

---

## SQL row order

The `ROW_NUMBER` and `COUNT` windows run through `window()`, which keeps input order. Both SQL backends break qbin ties by input position, tagged with a row-number helper column. See [SQL backends: window() keeps row order](02-row-preserving-contract.md#sql-backends-window-keeps-row-order). Pandas and PyArrow assign by index and need no workaround.

---

## Usage

```python
from mloda.user import Feature, PluginLoader, mloda

PluginLoader.all()

data = {"value_int": [10, -5, 0, 20, 50, 30, 60, 15]}

features = [
    Feature("value_int__bin_5"),  # equal-width, 5 bins
    Feature("value_int__qbin_4"),  # quartiles
]

result = mloda.run_all(features, compute_frameworks=["PandasDataFrame"], api_data={"Src": data})
```

Row count matches the input; each new column contains integers in `[0, N-1]` (or NULL for unbinnable rows).

---

## Related

- [Row-preserving contract](02-row-preserving-contract.md) - How SQL `window()` keeps input order and when a tag-and-restore wrapper is still needed.
- [Reference implementation pattern](03-reference-implementation.md) - the pure Python reference (`reference_binning`) is the source of truth.
- [Adding a new data operation](10-adding-new-operation.md) - Template for extending binning to a new framework.
