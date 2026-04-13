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
8. [Scalar and frame aggregate](08-scalar-and-frame-aggregate.md) - Global broadcast and rolling/expanding windows
9. [String operations](09-string-operations.md) - Element-wise string transforms
10. [Adding a new data operation](10-adding-new-operation.md) - End-to-end recipe
11. [Known divergences](known-divergences.md) - Audited cases where a framework would diverge from the PyArrow reference, with the mitigation for each
