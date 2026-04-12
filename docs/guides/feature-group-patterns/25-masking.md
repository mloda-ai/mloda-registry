# Masking

Conditional aggregation that nulls out non-matching values instead of removing rows.

**What**: FilterMask applies conditions that set non-matching values to null while preserving all rows.
**When**: You need conditional aggregation (e.g. "sum of sales where category = 'X'") alongside full-row-count results.
**Why**: Unlike filters, which remove rows entirely, masks keep every row and null out non-matching values. Aggregation functions naturally skip nulls, producing conditional results without changing the row count.
**Where**: Aggregation, Scalar Aggregate, Window Aggregation, Frame Aggregate, and Percentile feature groups.
**How**: Pass a `mask` key in `Options(context={...})` with `(column, operator, value)` tuples.

## Masks vs Filters

| | Filters | Masks |
|---|---|---|
| **Row count** | Rows removed | All rows preserved |
| **Non-matching values** | Eliminated | Set to null |
| **Use case** | Subset data before computation | Conditional aggregation |
| **API** | `GlobalFilter` on `mloda.run_all()` | `Options(context={"mask": ...})` per feature |

Filters and masks serve different purposes. Use filters when you want fewer rows in the output. Use masks when you want all rows but need aggregation to consider only a subset of values.

---

## Mask Spec Format

A mask specification is a tuple of `(column, operator, value)`:

```python
# Single condition
mask = ("category", "equal", "X")

# Multiple conditions (AND-combined)
mask = [
    ("category", "equal", "X"),
    ("value_int", "greater_equal", 10),
]
```

Multiple conditions are combined with AND logic. All conditions must be true for a value to be included in the aggregation.

### Supported Operators

| Operator | Description | Example |
|---|---|---|
| `equal` | Exact match | `("category", "equal", "X")` |
| `greater_than` | Strict greater than | `("value_int", "greater_than", 10)` |
| `greater_equal` | Greater than or equal | `("value_int", "greater_equal", 10)` |
| `less_than` | Strict less than | `("amount", "less_than", 100)` |
| `less_equal` | Less than or equal | `("amount", "less_equal", 100)` |
| `is_in` | Membership in a list | `("region", "is_in", ["A", "C"])` |

### Value Constraints

- Values must be `None`, `bool`, `int`, `float`, or `str`.
- For `is_in`, the value must be a non-empty `list`, `tuple`, or `set`.
- A 2-element tuple like `("col", "equal")` is valid only for the `equal` operator and checks for NULL.

---

## Basic Usage

Pass a `mask` key in the feature's context options:

```python
from mloda.user import Feature, Options, mloda, PluginLoader

PluginLoader.all()

# Sum of value_int where category equals 'X', partitioned by region
feature = Feature(
    "value_int__sum_agg",
    Options(context={
        "partition_by": ["region"],
        "mask": ("category", "equal", "X"),
    }),
)

result = mloda.run_all(
    [feature],
    compute_frameworks={"PandasDataFrame"},
)
```

Rows where `category != 'X'` have their `value_int` set to null before the SUM aggregation. The aggregation skips nulls, so the result contains only the sum of values matching the condition.

---

## Multiple Conditions

Combine conditions in a list. All conditions are AND-combined:

```python
from mloda.user import Feature, Options

# Sum where category='X' AND value_int >= 10
feature = Feature(
    "value_int__sum_agg",
    Options(context={
        "partition_by": ["region"],
        "mask": [
            ("category", "equal", "X"),
            ("value_int", "greater_equal", 10),
        ],
    }),
)
```

Only rows satisfying all conditions contribute to the aggregate.

---

## Supported Feature Groups

All of the following feature groups accept the `mask` key in their context options:

| Feature Group | Naming Pattern | Example | Row Behavior |
|---|---|---|---|
| Aggregation | `{col}__{agg}_agg` | `value_int__sum_agg` | Reduces rows (one per partition) |
| Window Aggregation | `{col}__{agg}_window` | `value_int__sum_window` | Preserves rows (broadcasts per partition) |
| Scalar Aggregate | `{col}__{agg}_scalar` | `value_int__sum_scalar` | Preserves rows (broadcasts globally) |
| Frame Aggregate (rolling) | `{col}__{agg}_rolling_{N}` | `value_int__sum_rolling_3` | Preserves rows (rolling window) |
| Frame Aggregate (time window) | `{col}__{agg}_{size}_{unit}_window` | `value_int__avg_7_day_window` | Preserves rows (time-interval window) |
| Frame Aggregate (cumulative) | `{col}__cum{agg}` | `value_int__cumsum` | Preserves rows (running aggregate) |
| Frame Aggregate (expanding) | `{col}__expanding_{agg}` | `value_int__expanding_avg` | Preserves rows (expanding window) |
| Percentile | `{col}__p{N}_percentile` | `value_int__p95_percentile` | Preserves rows (broadcasts per partition) |

For all types, the mask is applied before the aggregation. Non-matching values become null, and the aggregation function skips nulls.

---

## Framework Behavior

Users select a framework via `compute_frameworks` and do not interact with mask internals directly. Each framework applies masks using its native approach:

| Framework | Mechanism |
|---|---|
| Pandas | Boolean Series via `.where(mask)`, non-matching values become NaN |
| PyArrow | Null replacement via `pc.if_else` on a boolean array |
| Polars (lazy) | `pl.when(mask).then(col).otherwise(None)` with a temporary column |
| DuckDB / SQLite | SQL `CASE WHEN condition THEN source END` expression |

The framework selection does not affect the mask spec format. The same `("column", "operator", value)` tuples work across all frameworks.

---

## Testing with MaskTestMixin

The testing library provides two mixins for verifying masking in custom feature groups:

- `MaskTestMixin` provides 6 unit-level test methods covering equal, multiple conditions, is_in, greater_than, fully masked, and no-mask baseline scenarios.
- `MaskIntegrationTestMixin` provides 3 pipeline-level test methods that verify masking through the full `mloda.run_all` pipeline.

Both mixins use overridable class methods for configuration:

```python
from mloda.testing.feature_groups.data_operations.mixins.mask import MaskTestMixin
from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase

class MyAggregationTestBase(MaskTestMixin, DataOpsTestBase):
    @classmethod
    def mask_feature_name(cls) -> str:
        return "value_int__sum_agg"

    @classmethod
    def mask_partition_by(cls) -> list[str] | None:
        return ["region"]

    @classmethod
    def mask_expected_row_count(cls) -> int:
        return 4  # One row per partition (reducing)

    @classmethod
    def mask_is_reducing(cls) -> bool:
        return True  # Aggregation reduces rows

    @classmethod
    def mask_equal_expected(cls) -> dict:
        return {"A": 10, "B": 60, "C": 15, "D": -10}

    # Override remaining methods: mask_multiple_conditions_expected,
    # mask_is_in_expected, mask_greater_than_expected, mask_no_mask_expected
```

Concrete test classes then compose the test base with a framework mixin:

```python
from mloda.testing.feature_groups.data_operations.mixins.pandas import PandasTestMixin

class TestPandasMyAggregation(PandasTestMixin, MyAggregationTestBase):
    @classmethod
    def implementation_class(cls):
        return PandasMyAggregation
```

This pattern ensures mask behavior is verified across all supported frameworks with consistent expected values.

---

## Related

- [Filter Concepts](15-filter-concepts.md) - Filters remove rows; masks null out values
- [Options](11-options.md) - Passing mask specs via context options
- [Testing Guide](10-testing-guide.md) - Testing levels for feature groups
