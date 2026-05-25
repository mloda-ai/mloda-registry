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
    USER  → Require via data_connections parameter

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

## Column Data Type Extraction

Compute framework plugins should implement `_extract_column_data_type(data, column_name) -> DataType | None`
when they can inspect native column types. `return_data_type_rule()` declares the expected feature output type,
and mloda uses `_extract_column_data_type()` to validate that expectation against the actual framework result.
Without this hook, strict `return_data_type_rule` checks cannot verify plugin-authored framework outputs end-to-end.

Return a concrete `DataType` when the framework exposes enough precision to map the native type reliably. Return
`None` when the column is missing, schema inspection is unavailable without executing an unsafe or expensive query,
or the native type system is too ambiguous to validate. For collapsed native types, return the closest safe
mloda type only when that is stable for the framework; otherwise return `None`.

```python
from typing import Any

from mloda.user import DataType


def _extract_column_data_type(self, data: Any, column_name: str) -> DataType | None:
    if column_name not in data.columns:
        return None

    dtype = data[column_name].dtype

    if is_int32_dtype(dtype):
        return DataType.INT32
    if is_integer_dtype(dtype):
        return DataType.INT64
    if is_float32_dtype(dtype):
        return DataType.FLOAT
    if is_float_dtype(dtype):
        return DataType.DOUBLE
    if is_bool_dtype(dtype):
        return DataType.BOOLEAN
    if is_string_dtype(dtype):
        return DataType.STRING

    return None
```

Common mapping decisions:

- Use the narrowest exact type when the framework exposes precision, such as `int32` to `DataType.INT32` and
  `int64` to `DataType.INT64`.
- Use safe widening when the framework collapses widths, such as an integer-only type system returning
  `DataType.INT64` for integer columns.
- For timestamp systems with one native timestamp type, such as Spark or Iceberg, document the chosen
  `TIMESTAMP_MILLIS` or `TIMESTAMP_MICROS` mapping and keep it consistent with the framework's runtime behavior.
- For affinity-based systems such as SQLite, return a `DataType` only when the concrete value or declared type is
  reliable enough; otherwise return `None` so validation does not claim false precision.

Short collapsed-precision example:

```python
def _extract_column_data_type(self, data: Any, column_name: str) -> DataType | None:
    native_type = data.schema[column_name]

    if native_type == "INTEGER":
        return DataType.INT64  # Framework does not distinguish INT32 from INT64.
    if native_type == "TIMESTAMP":
        return DataType.TIMESTAMP_MICROS  # Chosen framework convention.

    return None
```
