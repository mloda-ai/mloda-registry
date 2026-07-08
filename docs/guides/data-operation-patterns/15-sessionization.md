# Sessionization

Sessionization groups a stream of timestamped events into sessions and assigns each row a session id. A new session starts whenever the gap to the previous event (in time order, within the same partition) exceeds a threshold. This is the standard inactivity-timeout rule used in recsys, telemetry, and clickstream pipelines.

**What**: `SessionizationFeatureGroup` accepts feature names of the form `{ts}__sessionize_{n}_{unit}`, where `n` is a positive integer and `unit` is one of `minute` / `hour` / `day` / `week`. It reads optional `partition_by` and `order_by` from `Options(context=...)` and emits a new `{ts}__sessionize_{n}_{unit}` integer column.
**When**: You want to segment per-user (or per-entity) event streams into visits or sessions for downstream features such as session length, events-per-session, or within-session ordering.
**Why**: The rule (sort, diff consecutive timestamps, flag gaps over a threshold, cumulative-sum the flags) is mechanical but easy to get subtly wrong across backends (boundary handling, partition isolation, row-order preservation, floating-point gap math on SQLite). The cross-framework contract pins one semantic.
**Where**: `mloda/community/feature_groups/data_operations/row_preserving/sessionization/`.
**How**: The feature name carries the threshold; the ordering timestamp and partitions come from Options. Rows are sorted by `[*partition_by, order_by]`, a new-session flag is computed per partition in time order, and a globally-unique session id is the cumulative count of those flags. The original row order is restored on output.

---

## Pattern

```text
ts__sessionize_30_minute
ts__sessionize_1_hour
```

| Option | Required | Meaning |
|---|---|---|
| `order_by` | no (defaults to the source timestamp) | The timestamp column rows are ordered by within each partition. |
| `partition_by` | no (default `[]`) | Columns that scope sessions (e.g. `["user_id"]`). With no partition the whole table is one stream. |
| `in_features` | config form only | The source timestamp column (when not using the `{ts}__sessionize_{n}_{unit}` string form). |

The threshold is `n` of `unit` as a fixed duration: `minute` = 60s, `hour` = 3600s, `day` = 86400s, `week` = 604800s. Only fixed-duration units are supported; calendar units (`month`, `year`) are intentionally excluded because a "gap" is a duration and those units are not fixed-length.

---

## Semantics (pinned)

Within each partition, rows are sorted by `order_by` ascending. A row **starts a new session** if it is the first row in its partition, or the gap to the previous row is **strictly greater** than the threshold:

```text
is_new[i] = first-in-partition[i] OR (ts[i] - ts[prev] > threshold)
session_id = cumsum(is_new) - 1        # over the [partition, ts]-sorted frame
```

- A gap **equal** to the threshold stays in the **same** session (the comparison is strict `>`).
- Session ids are **globally unique** 0-based integers, not reset per partition: distinct partitions never share a session id, so you can group by the session id alone.
- **Row-preserving**: same rows, same original order, one new integer column; the source timestamp column is unchanged.

Example with a 30-minute threshold for one user whose events are at `10:00, 10:20, 10:50, 11:30, 11:35`:

| ts | gap from prev | session |
|---|---|---|
| 10:00 | (first) | 0 |
| 10:20 | 20 min | 0 |
| 10:50 | 30 min (== threshold) | 0 |
| 11:30 | 40 min (> threshold) | 1 |
| 11:35 | 5 min | 1 |

---

## Backend support: all five native

| Backend | Behavior |
|---|---|
| PyArrow | native: sort indices, sliced timestamp diff, `cumulative_sum`, scatter back. The reference oracle. |
| Pandas | native `groupby(partition_by)[ts].diff()` + `cumsum`. |
| Polars (lazy) | native `diff().over(partition_by)` + `cum_sum`. |
| DuckDB | native SQL: `LAG` + `date_diff('second', ...)` over a window, then a running `SUM`. |
| SQLite | native SQL window functions: `LAG` + `ROUND((julianday(b) - julianday(a)) * 86400.0)` for the gap, then a running `SUM`. |

The gap-diff plus cumulative-sum is expressible natively on every backend, so none rejects the operation. The SQLite backend rounds the `julianday` gap to whole seconds so floating-point noise cannot push an exact-threshold gap over the boundary.

---

## Usage

```python
from mloda.user import Feature, Options, PluginLoader, mloda

PluginLoader.all()

features = [
    Feature("ts__sessionize_30_minute", Options(context={"partition_by": ["user_id"]})),
]
result = mloda.run_all(features, compute_frameworks={"PandasDataFrame"})
```

---

## Related

- [Row-preserving contract](02-row-preserving-contract.md) - Output row count and order must match input.
- [EMA](13-ema.md) - The other ordered, partitioned row-preserving time-series op.
- [Forward fill by time](12-ffill-by-time.md) - Carry the last non-null value forward across time gaps, per partition.
- [Time bucketization](11-time-bucketization.md) - Map a timestamp onto coarser interval boundaries.
