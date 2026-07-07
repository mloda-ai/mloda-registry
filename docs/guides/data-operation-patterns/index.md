# Data Operation Patterns

Data operations are built-in feature groups that transform existing columns: row-preserving analytics (binning, ranks, windows, offsets, percentiles), group-reducing aggregations, and element-wise string transforms.

These guides describe the contracts each category must honor, the cross-framework reference-implementation pattern, and the end-to-end recipe for adding a new operation.

Read them after you are comfortable with the [feature-group patterns](../feature-group-patterns/01-root-features.md). They extend that model with the additional invariants that make a single feature name produce identical results on PyArrow, Pandas, Polars, DuckDB, and SQLite.

## Guides

1. [Overview](01-overview.md) - Categories, naming patterns, and where the code lives
2. [Row-preserving contract](02-row-preserving-contract.md) - Output row count and order must match input
3. [Reference implementation pattern](03-reference-implementation.md) - PyArrow is the source of truth
4. [Supported ops per framework](04-supported-ops.md) - Excluding ops a framework cannot express
5. [Binning](05-binning.md) - `bin` vs `qbin`, NTILE vs rank, NULL handling
6. [Window aggregation](06-window-aggregation.md) - Partitioned aggregates broadcast per row
7. [Percentile, rank, offset](07-percentile-rank-offset.md) - The analytic window family
8. [Scalar aggregate, frame aggregate, scalar arithmetic, and point arithmetic](08-scalar-and-frame-aggregate.md) - Global broadcast, rolling/expanding windows, element-wise column-vs-constant arithmetic, and element-wise column-vs-column arithmetic
9. [String operations](09-string-operations.md) - Element-wise string transforms
10. [Adding a new data operation](10-adding-new-operation.md) - End-to-end recipe
11. [Time bucketization](11-time-bucketization.md) - `floor` / `ceil` / `round` a timestamp to a bucket interval (minute / hour / day / week / month / year)
12. [Forward fill by time](12-ffill-by-time.md) - Carry the last non-null value forward across time gaps, per partition (row-preserving)
13. [EMA](13-ema.md) - Exponential moving average (`{col}__ema_{span}`), per partition; pandas / polars native, pyarrow / duckdb / sqlite not implemented (no backend)
14. [Resample](14-resample.md) - Collapse events onto a regular time grid (`{col}__resample_{n}_{unit}_{agg}`); the first `row_changing` operation
15. [Sessionization](15-sessionization.md) - Assign a gap-threshold session id on an ordered timestamp (`{ts}__sessionize_{n}_{unit}`), per partition (row-preserving)
16. [`return_data_type_rule` failure handling](16-return-data-type-rule.md) - The fail-fast, post-selection contract: why the type rules no longer catch extraction errors, how binning/resample matching was tightened so a selected feature never raises, and the completeness-test guard (#244 -> #265, core #485/#493)
17. [Known divergences](known-divergences.md) - Audited cases where a framework would diverge from the PyArrow reference, with the mitigation for each
18. [Framework support matrix](framework-support-matrix.md) - Operation x framework capability table, generated from `supported_*()` test overrides
