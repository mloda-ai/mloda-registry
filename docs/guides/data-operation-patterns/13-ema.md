# Exponential moving average (EMA)

EMA computes an exponentially weighted moving average of a column in time order, giving recent observations more weight than older ones. It is also called exponential decay or EWMA.

**What**: `EmaFeatureGroup` accepts feature names of the form `{col}__ema_{span}`, where `span` is a positive integer. It reads `order_by` and optional `partition_by` from `Options(context=...)` and emits a new `{col}__ema_{span}` column.
**When**: You want a smoothed, recency-weighted signal (moving average of a price, decaying activity score, smoothed sensor reading) for recsys, telemetry, or any time-series pipeline.
**Why**: EMA is a sequential recurrence with no native primitive on PyArrow or the SQL engines, and pandas / polars disagree by default on null handling. The cross-framework contract pins one semantic.
**Where**: `mloda/community/feature_groups/data_operations/row_preserving/ema/`.
**How**: The feature name carries the span; the time column and partitions come from Options. The recurrence runs in time order within each partition; the original row order is restored on output.

---

## Pattern

```text
value__ema_2
value__ema_3
```

| Option | Required | Meaning |
|---|---|---|
| `order_by` | yes | The time column. Rows are sorted ascending within each partition before the recurrence. |
| `partition_by` | no (default `[]`) | Columns that scope the EMA. Each partition has an independent recurrence. |
| `in_features` | config form only | The source column (when not using the `{col}__ema_{span}` string form). |

`span` maps to the smoothing factor `alpha = 2 / (span + 1)`. The span is passed straight to the underlying library so pandas and polars apply the identical mapping.

---

## Semantics (pinned)

EMA is defined with `adjust=False` and **nulls skipped in the recurrence**:

```text
ema[i] = alpha * x[i] + (1 - alpha) * ema[i-1]      # for non-null x[i]
ema[i] = null                                         # where x[i] is null
```

- The first non-null value seeds the recurrence (`ema[seed] = x[seed]`).
- A null **input** produces a null **output**, and does not advance the recurrence: the next non-null value is combined with the last non-null EMA state, not with a null.
- **Row-preserving**: same rows, same original order, one new column; the source column is unchanged.

This is exactly: pandas `series.ewm(span=SPAN, adjust=False, ignore_na=True).mean().mask(series.isna())` ≡ polars `pl.col(c).ewm_mean(span=SPAN, adjust=False, ignore_nulls=True)`. The two were verified to agree element-wise (including null positions) before the spec was pinned.

---

## Backend support: pandas and polars only

| Backend | Behavior |
|---|---|
| Pandas | native `ewm(span=..., adjust=False, ignore_na=True)`, null-masked. |
| Polars (lazy) | native `ewm_mean(span=..., adjust=False, ignore_nulls=True)` over the partition. |
| PyArrow | not implemented (no backend). PyArrow compute has no exponential-weighted primitive. |
| DuckDB | not implemented (no backend). No native EWM. |
| SQLite | not implemented (no backend). No native EWM. |

PyArrow, DuckDB, and SQLite ship no EMA backend rather than emulating the recurrence row-by-row in Python, following the project's CFW backend rule and the same absence convention other data operations use (for example resample has no SQLite backend). A request that resolves only to one of these frameworks fails with mloda core's generic no-feature-group error; where pandas or polars-lazy is available, EMA resolves there.

---

## Usage

```python
from mloda.user import Feature, Options, PluginLoader, mloda

PluginLoader.all()

features = [
    Feature("price__ema_10", Options(context={"order_by": "ts", "partition_by": ["symbol"]})),
]
result = mloda.run_all(features, compute_frameworks={"PandasDataFrame"})
```

---

## Related

- [Row-preserving contract](02-row-preserving-contract.md) - Output row count and order must match input.
- [Supported ops per framework](04-supported-ops.md) - How a framework declares (or rejects) an op it cannot express.
- [Forward fill by time](12-ffill-by-time.md) - The other ordered, partitioned row-preserving time-series op.
