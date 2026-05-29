# Concept 10: Data Type Extraction

Compute framework plugins expose their native column types to mloda through `_extract_column_data_type()`.

**What**: Hook that maps a framework's native column type to an mloda `DataType`.
**When**: Implement when the framework can inspect native column types.
**Why**: `return_data_type_rule()` declares the expected feature output type; mloda uses this hook to validate that expectation against the actual framework result.
**Where**: Defined on the compute framework class as `_extract_column_data_type(self, data, column_name) -> DataType | None`.
**How**: Inspect the native column type and return the closest safe `DataType`, or `None` when validation cannot be trusted.

## Signature

```python
def _extract_column_data_type(self, data: Any, column_name: str) -> DataType | None: ...
```

`return_data_type_rule()` declares the expected feature output type, and mloda uses
`_extract_column_data_type()` to validate that expectation against the actual framework result.
Without this hook, strict `return_data_type_rule` checks cannot verify plugin-authored framework
outputs end-to-end.

## When to Return a Type vs None

Return a concrete `DataType` when the framework exposes enough precision to map the native type
reliably. Return `None` when:

- the column is missing,
- schema inspection is unavailable without executing an unsafe or expensive query,
- or the native type system is too ambiguous to validate.

For collapsed native types, return the closest safe mloda type only when that mapping is stable for
the framework; otherwise return `None`.

## Example

The predicates below (`_is_int32`, `_is_integer`, ...) are framework-supplied placeholders. Replace
them with your library's own type checks.

```python
from typing import Any

from mloda.user import DataType


def _extract_column_data_type(self, data: Any, column_name: str) -> DataType | None:
    if column_name not in data.columns:
        return None

    dtype = data[column_name].dtype

    if _is_int32(dtype):
        return DataType.INT32
    if _is_integer(dtype):
        return DataType.INT64
    if _is_float32(dtype):
        return DataType.FLOAT
    if _is_float(dtype):
        return DataType.DOUBLE
    if _is_bool(dtype):
        return DataType.BOOLEAN
    if _is_string(dtype):
        return DataType.STRING

    return None
```

## Common Mapping Decisions

- Use the narrowest exact type when the framework exposes precision, such as `int32` to
  `DataType.INT32` and `int64` to `DataType.INT64`.
- Use safe widening when the framework collapses widths, such as an integer-only type system
  returning `DataType.INT64` for integer columns.
- For timestamp systems with one native timestamp type, such as Spark or Iceberg, document the
  chosen `TIMESTAMP_MILLIS` or `TIMESTAMP_MICROS` mapping and keep it consistent with the
  framework's runtime behavior.
- For affinity-based systems such as SQLite, return a `DataType` only when the concrete value or
  declared type is reliable enough; otherwise return `None` so validation does not claim false
  precision.

## Collapsed-Precision Example

```python
def _extract_column_data_type(self, data: Any, column_name: str) -> DataType | None:
    native_type = data.schema[column_name]

    if native_type == "INTEGER":
        return DataType.INT64  # Framework does not distinguish INT32 from INT64.
    if native_type == "TIMESTAMP":
        return DataType.TIMESTAMP_MICROS  # Chosen framework convention.

    return None
```
