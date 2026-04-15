# Data Operations

Built-in feature groups that transform existing columns: row-preserving analytics, row-reducing aggregations, and element-wise string transforms. A single feature name produces the same result on PyArrow, Pandas, Polars, DuckDB, and SQLite.

## Categories

| Category | Location | Row behavior | Examples |
|---|---|---|---|
| Row-preserving | `row_preserving/` | Output row count and order match input | binning, window aggregation, rank, offset, percentile, scalar aggregate, frame aggregate, datetime |
| Aggregation | `aggregation/` | Reduces to one row per group | sum, avg, count, min, max, std, var, median, mode, nunique, first, last |
| String | `string/` | Row-preserving, element-wise on strings | upper, lower, trim, length, reverse |

## Documentation

Full guides live in [`docs/guides/data-operation-patterns/`](../../../../docs/guides/data-operation-patterns/index.md):

1. [Overview](../../../../docs/guides/data-operation-patterns/01-overview.md) - categories, naming patterns, code locations
2. [Row-preserving contract](../../../../docs/guides/data-operation-patterns/02-row-preserving-contract.md)
3. [Reference implementation pattern](../../../../docs/guides/data-operation-patterns/03-reference-implementation.md) - PyArrow is the source of truth
4. [Supported ops per framework](../../../../docs/guides/data-operation-patterns/04-supported-ops.md)
5. [Binning](../../../../docs/guides/data-operation-patterns/05-binning.md)
6. [Window aggregation](../../../../docs/guides/data-operation-patterns/06-window-aggregation.md)
7. [Percentile, rank, offset](../../../../docs/guides/data-operation-patterns/07-percentile-rank-offset.md)
8. [Scalar and frame aggregate](../../../../docs/guides/data-operation-patterns/08-scalar-and-frame-aggregate.md)
9. [String operations](../../../../docs/guides/data-operation-patterns/09-string-operations.md)
10. [Adding a new data operation](../../../../docs/guides/data-operation-patterns/10-adding-new-operation.md)
11. [Known divergences](../../../../docs/guides/data-operation-patterns/known-divergences.md)
