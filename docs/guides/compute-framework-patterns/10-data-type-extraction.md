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

## Validation and Mismatches

After a feature is computed, mloda's `DataTypeValidator` compares the feature's declared
`data_type` (from `return_data_type_rule()`) against the type your hook reports for the produced
column. Validation is **skipped** whenever the feature declares no `data_type` or your hook returns
`None`, so a framework that never overrides the hook silently performs no enforcement.

When both types are present, a mismatch raises `DataTypeMismatchError`:

```text
Feature 'price': declared STRING, got INT64, coercion not supported
```

Compatibility is checked in one of two modes, selected per run via `strict_type_enforcement` on
`mloda.run_all` (default `False`):

| Mode | Flag | Rule |
|------|------|------|
| Lenient (default) | `strict_type_enforcement=False` | Any numeric type (`INT32`, `INT64`, `FLOAT`, `DOUBLE`) is interchangeable with any other numeric type; any timestamp type is interchangeable with any other timestamp type; every other type must match exactly. |
| Strict | `strict_type_enforcement=True` | Only safe widening is allowed — a declared type accepts a narrower actual type: declared `INT64` accepts actual `INT32`; declared `DOUBLE` accepts actual `INT32`/`INT64`/`FLOAT`; declared `TIMESTAMP_MICROS` accepts actual `TIMESTAMP_MILLIS`. Every other pairing, including narrowing, must match exactly. |

### Worked Example

A feature declares `DataType.INT32` and the framework produces a `DOUBLE` column:

```python
result = mloda.run_all(features)                              # lenient: INT32 vs DOUBLE → OK (both numeric)
result = mloda.run_all(features, strict_type_enforcement=True)  # strict: DOUBLE is a narrowing of INT32 → DataTypeMismatchError
```

Because your hook is the only thing that reports the *actual* produced type, returning an accurate
`DataType` is what makes strict end-to-end enforcement possible; returning `None` opts the column out
of the check entirely.

## Real Implementations

| File | Description |
|------|-------------|
| [compute_framework.py](https://github.com/mloda-ai/mloda/blob/main/mloda/core/abstract_plugins/compute_framework.py) | `_extract_column_data_type` hook + `run_validate_output_features` |
| [datatype_validator.py](https://github.com/mloda-ai/mloda/blob/main/mloda/core/abstract_plugins/components/validators/datatype_validator.py) | `DataTypeValidator`, `DataTypeMismatchError`, strict/lenient rules |
| [python_dict_framework.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/python_dict/python_dict_framework.py) | Canonical minimal `_extract_column_data_type` override |

See the companion core-docs ticket [mloda-ai/mloda#465](https://github.com/mloda-ai/mloda/issues/465)
for the conceptual reference on the hook and `return_data_type_rule` enforcement.
