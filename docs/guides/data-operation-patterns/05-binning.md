# Binning

Binning maps a numeric column onto integer bucket indices `0..n-1`. Two variants ship: equal-width (`bin`) and quantile-based (`qbin`).

**What**: `BinningFeatureGroup` accepts feature names of the form `{col}__bin_{N}` or `{col}__qbin_{N}`.
**When**: You need discretization for downstream grouping, charting, or model features.
**Why**: Both operations are standard, but implementing them identically across frameworks is non-trivial because of NTILE reordering and NULL propagation.
**Where**: `mloda/community/feature_groups/data_operations/row_preserving/binning/`.
**How**: Feature name encodes the op and bin count; the framework implementation computes the bin index per row and preserves input order.

---

## Two binning modes

```python
# mloda/community/feature_groups/data_operations/row_preserving/binning/base.py
BINNING_OPS = {
    "bin":  "Equal-width binning (value range divided into n equal intervals)",
    "qbin": "Quantile-based binning (rows divided into n roughly equal groups by rank)",
}
```

| Mode | Semantics | Result shape |
|---|---|---|
| `bin` (equal-width) | Divide `[min, max]` into N intervals of equal width. Bin index = `floor((value - min) / width)`, clamped to `N-1`. | Bins may be unbalanced if the distribution is skewed. |
| `qbin` (quantile) | Sort rows by value, assign rank `r` to bin `r * N // n` where `n` is the count of non-null values. | Bins are always roughly equal in row count. |

`qbin` uses rank-based assignment rather than sample quantiles. That is a deliberate choice to sidestep interpolation disagreements across frameworks: ranks are integers and leave no room for numerical drift.

---

## NTILE vs rank-based equivalence

SQL engines offer `NTILE(N)` to partition rows into N buckets. Its semantics are "assign bucket `ceil(rank * N / n)`", which is equivalent to the rank-based formula `r * N // n` up to a 1-based-vs-0-based offset. The mloda convention is 0-based, so SQL implementations subtract 1 and clamp at `N-1`.

Pseudocode equivalence:

| Expression | Produces |
|---|---|
| PyArrow `rank * n_bins // n` | 0..N-1 |
| DuckDB `NTILE(N) OVER (ORDER BY col) - 1` | 0..N-1 |
| Pandas `(rank * N // n).astype("Int64")` | 0..N-1 |

All three resolve to the same labels for the same row order, but only after accounting for how each framework handles ties and NULLs.

---

## NULL and NaN handling

- Input NULLs are skipped when computing `min`, `max`, and the non-null count `n`.
- Rows with NULL in the source column receive NULL in the bin column, not a real bin index.
- NaN in floating columns is treated like NULL. DuckDB explicitly guards with `isnan(col)`:

```sql
CASE WHEN col IS NULL OR isnan(col) THEN NULL
     ELSE LEAST(NTILE(N) OVER (
         PARTITION BY CASE WHEN col IS NOT NULL AND NOT isnan(col) THEN 1 END
         ORDER BY col) - 1, N - 1) END
```

The `PARTITION BY CASE WHEN ...` clause is what ensures NULL/NaN rows do not participate in the rank at all; they stay unpartitioned and the CASE returns NULL for them.

---

## The DuckDB row-order workaround

`NTILE` requires `ORDER BY col`, which produces rows sorted by `col`, not by original input order. The row-preserving contract requires input order. The fix:

```python
# Paraphrased from duckdb_binning.py
qrn = quote_ident("__mloda_rn__")
with_rn    = data.select(_raw_sql=f"*, ROW_NUMBER() OVER () AS {qrn}")
with_qbin  = with_rn.select(_raw_sql=f"*, {ntile_expression} AS {quoted_feature}")
sorted_rel = with_qbin.order(qrn)
# project out __mloda_rn__ in the final select
```

1. Tag each input row with `ROW_NUMBER() OVER ()` before the window runs.
2. Apply the NTILE-based expression.
3. Re-sort by the tagged row number to restore input order.
4. Drop the temporary column from the projection.

Any framework whose native operation reorders will need an analogous workaround. Pandas and PyArrow do not, because both assign by index directly. See [the row-preserving contract](02-row-preserving-contract.md) for the general rule.

---

## Usage

```python
from mloda.user import Feature, PluginLoader, mloda

PluginLoader.all()

features = [
    Feature("value_int__bin_5"),    # equal-width, 5 bins
    Feature("value_int__qbin_4"),   # quartiles
]

result = mloda.run_all(features, compute_frameworks={"PandasDataFrame"})
```

Row count matches the input; each new column contains integers in `[0, N-1]` (or NULL for unbinnable rows).

---

## Related

- [Row-preserving contract](02-row-preserving-contract.md) - Why the DuckDB `ROW_NUMBER()` tag-and-restore pattern exists.
- [Reference implementation pattern](03-reference-implementation.md) - PyArrow's rank-based `qbin` is the source of truth.
- [Adding a new data operation](10-adding-new-operation.md) - Template for extending binning to a new framework.
