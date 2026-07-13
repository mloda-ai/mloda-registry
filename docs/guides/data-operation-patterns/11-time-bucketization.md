# Time bucketization

Time bucketization maps a timestamp column onto coarser interval boundaries. Three modes ship: `floor` (round down), `ceil` (round up), and `round` (round to nearest, half-up).

**What**: `TimeBucketizationFeatureGroup` accepts feature names of the form `{src}__{op}_{n}_{unit}`, where `op` is `floor` / `ceil` / `round`, `n` is a positive integer bucket size, and `unit` is one of `minute` / `hour` / `day` / `week` / `month` / `year`.
**When**: You need to align timestamps to a common interval for grouping, sessionization, or feature engineering (e.g. "events per hour", "first of the month", "ISO week").
**Why**: Doing this consistently across PyArrow, Pandas, Polars, DuckDB, and SQLite is harder than it looks: tie-breaking, week-start convention, calendar-unit ceil semantics, and timestamp-type fidelity all diverge by default.
**Where**: `mloda/community/feature_groups/data_operations/row_preserving/time_bucketization/`.
**How**: Feature name encodes the op, bucket size, and unit; the framework implementation computes the bucket-aligned timestamp per row and preserves input order, tz, and resolution.

---

## Three bucketization modes

```python
# mloda/community/feature_groups/data_operations/row_preserving/time_bucketization/base.py
TIME_BUCKETIZATION_OPS = {
    "floor": "Round timestamp down to the start of the enclosing bucket",
    "ceil":  "Round timestamp up to the start of the next bucket (idempotent on aligned for fixed-freq)",
    "round": "Round timestamp to the nearest bucket boundary (half rounds up)",
}
```

| Mode | Semantics |
|---|---|
| `floor` | Snap to the start of the enclosing bucket. `floor(14:37, 5_minute) = 14:35`. |
| `ceil` | Snap to the start of the next bucket. Per-unit idempotency rules below. |
| `round` | Pick whichever of `floor` / `ceil` is closer. Half-way values always go UP. |

---

## Pattern and bucket size

Feature names follow `{src}__{op}_{n}_{unit}` with an explicit `n` (no implicit `_1_`). Examples:

```text
timestamp__floor_1_day
timestamp__ceil_15_minute
timestamp__round_1_hour
timestamp__floor_1_week
```

| Unit | Supported `n` |
|---|---|
| `minute`, `hour`, `day` | Any positive integer |
| `week`, `month`, `year` | Only `n=1` (other values raise `ValueError`) |

`n > 1` is rejected for calendar units because their length is not uniform (months have 28-31 days; years differ by leap day). A non-aligned multi-week / multi-month / multi-year bucket would be epoch-anchored, which is rarely what users intend. The constraint is enforced in `_parse_bucket_op` and surfaces at compute time.

---

## Week start: ISO Monday

`week` floors to the most recent Monday at midnight. This is the ISO 8601 convention and matches PyArrow's `floor_temporal(..., week_starts_monday=True)`. The most surprising case is the transition around new year:

```python
# Sun 2023-01-01 → Mon 2022-12-26 (the previous Monday)
# Mon 2023-01-02 → Mon 2023-01-02 (idempotent)
```

Pandas' default `Period` alias for "week starting Monday" is `W-SUN` (week *ending* Sunday); the FG uses that alias internally. DuckDB's `DATE_TRUNC('week', col)` is already Monday-anchored. Polars' `dt.truncate('1w')` is also Monday-anchored. SQLite has no native week truncation, so the implementation computes `date(col, '-N days')` where `N = (weekday + 6) % 7`.

---

## Round tie-breaking: half-up

Every midpoint rolls toward the *next* (higher) bucket. This matches PyArrow's `round_temporal` and Polars' `dt.round` defaults; the other backends (Pandas, DuckDB, SQLite) implement explicit midpoint comparisons to match.

| Input | Bucket | Output |
|---|---|---|
| `14:25:00` | 10 minute | `14:30:00` |
| `14:35:00` | 10 minute | `14:40:00` |
| `14:45:00` | 10 minute | `14:50:00` |
| `00:30:00` | 1 hour | `01:00:00` |

Pandas' built-in `Series.dt.round` is banker's rounding (half-to-even), so the FG bypasses it and computes `floor` / `ceil` directly, then picks via midpoint comparison.

---

## Ceil idempotency rule (PyArrow quirk)

`ceil_temporal(ceil_is_strictly_greater=False)` has unit-dependent behaviour pinned across all backends:

| Unit family | Aligned input behaviour |
|---|---|
| Fixed-freq (`minute`, `hour`, `day`) | Idempotent. `ceil(2023-01-01, 1_day) = 2023-01-01`. |
| Calendar (`week`, `month`, `year`) | Always advances. `ceil(2023-01-01, 1_year) = 2024-01-01`. |

For the day case, the test fixture's day-aligned rows (00:00:00) return themselves under `ceil_1_day`. For the year case, the same year-aligned `2023-01-01` returns `2024-01-01`. Every backend reproduces this quirk; DuckDB / SQLite / Pandas / Polars special-case the calendar units to always offset by one bucket.

---

## NULL propagation

A NULL timestamp produces NULL in the result. All backends propagate naturally:

- PyArrow `*_temporal` functions return null on null input.
- Pandas `dt.floor` / `dt.ceil` / `dt.round` preserve `NaT`.
- Polars expressions propagate null through `dt.truncate` / `dt.round` / `dt.offset_by`.
- DuckDB and SQLite use explicit `CASE WHEN col IS NULL THEN NULL ELSE ... END` guards in every CASE-based projection.

The Pandas backend additionally `mask`s the result column to convert `NaT` to `None` after the half-up round computation, since the expression-based round emits `NaT` for null inputs and the test contract compares to `None`.

---

## Timezone fidelity

Input timezone is preserved on output for every backend. The fixture used by the inherited tests is UTC, but a `pa.timestamp("us", tz="Europe/Berlin")` input flows through unchanged. Pandas' `PeriodIndex` strips tz, so `_calendar_floor` re-localises after `to_timestamp()`.

For SQLite, all timestamps are stored as TEXT in ISO 8601. The bucketization SQL emits ISO 8601 strings directly (bucket math on the wall-clock portion, timezone suffix reattached), so the result column comes back from `to_arrow_table()` as `pa.string()`, on a plain `SqliteRelation`. There is no Python-side re-parse into `pa.timestamp` and no `SqliteRelation` subclass; see `tests/test_sqlite_result_type.py` for the contract.

---

## Usage

```python
from mloda.user import Feature, PluginLoader, mloda

PluginLoader.all()

features = [
    Feature("event_time__floor_1_hour"),     # bucket events per hour
    Feature("event_time__floor_1_week"),     # ISO week start (Monday)
    Feature("event_time__ceil_1_day"),       # next-day boundary (idempotent on midnight)
    Feature("event_time__round_5_minute"),   # nearest 5-minute boundary, half-up
]

result = mloda.run_all(features, compute_frameworks={"PyArrowTable"})
```

Row count matches the input; each new column has the same timestamp type (resolution and tz) as the source.

---

## Related

- [Row-preserving contract](02-row-preserving-contract.md) - Output row count and order must match input.
- [Reference implementation pattern](03-reference-implementation.md) - PyArrow's `floor_temporal` / `ceil_temporal` / `round_temporal` are the cross-framework reference for time bucketization.
- [Adding a new data operation](10-adding-new-operation.md) - Template for extending time bucketization to a new framework or new unit.
