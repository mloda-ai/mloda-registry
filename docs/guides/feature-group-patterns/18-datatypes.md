# DataTypes

How to define and validate data types for features.

**What**: Arrow-based type system for features with runtime validation.
**When**: You want type safety and validation for feature outputs.
**Why**: Catch type mismatches early; ensure consistent data types across frameworks.
**Where**: Feature creation, FeatureGroup output declaration, validation.

## Available Types

| DataType | Description | Example |
|----------|-------------|---------|
| `INT32` | 32-bit integer | 42 |
| `INT64` | 64-bit integer | 9223372036854775807 |
| `FLOAT` | 32-bit float | 3.14 |
| `DOUBLE` | 64-bit float | 3.14159265359 |
| `BOOLEAN` | Boolean | True/False |
| `STRING` | UTF-8 string | "hello" |
| `BINARY` | Raw bytes | b"data" |
| `DATE` | Date | 2024-01-29 |
| `TIMESTAMP_MILLIS` | Timestamp (ms) | 2024-01-29T10:30:00.000 |
| `TIMESTAMP_MICROS` | Timestamp (μs) | 2024-01-29T10:30:00.000000 |
| `DECIMAL` | 128-bit decimal | 12345.123456789 |

## Typed Feature Constructors

```python
from mloda.user import Feature

Feature.int32_of("age")
Feature.int64_of("user_id")
Feature.double_of("price")
Feature.str_of("name")
Feature.boolean_of("is_active")
Feature.date_of("created_date")
Feature.timestamp_millis_of("event_time")
```

Or with explicit DataType:

```python
from mloda.user import Feature, DataType

Feature(name="age", data_type=DataType.INT32)
```

## Untyped Features

For backward compatibility or when type doesn't matter:

```python
Feature.not_typed("legacy_column")
```

## FeatureGroup Output Type

Declare fixed output type via `return_data_type_rule()`:

```python
from mloda.provider import FeatureGroup
from mloda.user import DataType, Feature

class UserCount(FeatureGroup):
    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> Optional[DataType]:
        return DataType.INT64
```

## Validation Modes

**Lenient (default):** Allows safe type widening (INT32→INT64, FLOAT→DOUBLE)

**Strict:** Exact type match required

```python
Feature.int32_of("exact_count", options={"strict_type_enforcement": True})
```

## Full Documentation

See [Data Type Enforcement](https://mloda-ai.github.io/mloda/in_depth/data-type-enforcement/) for detailed patterns.
