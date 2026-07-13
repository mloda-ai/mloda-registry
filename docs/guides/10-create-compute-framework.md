# Create a Compute Framework Plugin

Use this guide to add support for a new data processing library to mloda.

## Decision Tree

```text
Q1: Does your framework require a connection/session object?
    YES → Q2
    NO  → Q3

Q2: Is it a data lake table format (Iceberg, Delta, Hudi)?
    YES → Category 5: Data Lake
    NO  → Category 3: Stateful Connection

Q3: Does your framework use lazy evaluation?
    YES → Category 2: Stateless Lazy
    NO  → Q4

Q4: Does your framework have external dependencies?
    YES → Category 1: Stateless Eager
    NO  → Category 4: Zero Dependency

Q5: Do you need cross-framework conversion?
    YES → See 08-framework-transformer

Q6: Does your library have built-in PyArrow conversion?
    YES → Simplifies transformer (use .to_arrow()/.from_arrow())
    NO  → Manual conversion in 08-framework-transformer

Q7: Need to understand merge/join operations?
    YES → See 06-merge-engine

Q8: Do you need multi-column index support for joins?
    YES → See 06-merge-engine (Index with tuple)

Q9: Need to understand filter operations?
    YES → See 07-filter-engine

Q10: Should connections be auto-created or user-provided?
    AUTO  → Add fallback in set_framework_connection_object()
    USER  → Require via data_access_collection parameter

Q11: Ready to test your implementation?
    YES → See 09-testing-guide
```

## Category Guides

| Category | When to Use |
|----------|-------------|
| [01-stateless-eager](compute-framework-patterns/01-stateless-eager.md) | Simple in-memory frameworks (Pandas, PyArrow, Polars DataFrame) |
| [02-stateless-lazy](compute-framework-patterns/02-stateless-lazy.md) | Lazy evaluation frameworks (Polars LazyFrame, Ibis) |
| [03-stateful-connection](compute-framework-patterns/03-stateful-connection.md) | Connection/session required (DuckDB, SQLite, Spark) |
| [04-zero-dependency](compute-framework-patterns/04-zero-dependency.md) | Pure Python, no external libs (Python dict) |
| [05-data-lake](compute-framework-patterns/05-data-lake.md) | Catalog-based table formats (Iceberg, Delta) |

## Concepts

| Guide | What It Covers |
|-------|----------------|
| [06-merge-engine](compute-framework-patterns/06-merge-engine.md) | Join operations (INNER, LEFT, OUTER, APPEND, UNION) |
| [07-filter-engine](compute-framework-patterns/07-filter-engine.md) | Filter operations (range, equal, regex, categorical) |
| [08-framework-transformer](compute-framework-patterns/08-framework-transformer.md) | Cross-framework conversion (PyArrow as hub) |
| [09-testing-guide](compute-framework-patterns/09-testing-guide.md) | Testing your implementation |
| [10-data-type-extraction](compute-framework-patterns/10-data-type-extraction.md) | Mapping native column types to mloda `DataType` |

## Timezone Validation (Opt-In)

Merge and filter engines can opt in to a timezone guard that turns silent tz-aware vs tz-naive
mismatches into a clear `ValueError`.

- Opt in: set `provides_column_semantics = True` on your `BaseMergeEngine` / `BaseFilterEngine`
  subclass and implement `_column_semantics(data, column)` returning a `ColumnSemantics`.
- Default `False`: guard skipped, hook never required.
- Opted in without the hook: `NotImplementedError`.
- As-of joins require the hook regardless of the flag.

See [06-merge-engine](compute-framework-patterns/06-merge-engine.md#timezone-validation-opt-in)
and [07-filter-engine](compute-framework-patterns/07-filter-engine.md#timezone-validation-opt-in)
for details, and the upstream
[comparison contract](https://github.com/mloda-ai/mloda/blob/main/docs/docs/in_depth/comparison-contract.md)
for the full model.
