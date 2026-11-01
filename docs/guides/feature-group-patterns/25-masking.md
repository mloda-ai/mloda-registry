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
- For `is_in`, the value must be a non-empty `list`, `tuple`, `set`, or `frozenset`; a string raises `ValueError`.
- A 2-element tuple like `("col", "equal")` is valid only for the `equal` operator and matches null and NaN rows.

### Missing Values

A null or NaN row never passes a comparison (`equal` with a non-missing value, `greater_than`, `greater_equal`, `less_than`, `less_equal`), so its value is masked. The 2-element `equal` form and an `is_in` list containing `None` match null and NaN rows; a NaN value counts as `None`. Every framework follows this rule. See [Mask Engine](https://mloda-ai.github.io/mloda/in_depth/mask_engine/) for the core contract.

---

## Basic Usage

Pass a `mask` key in the feature's context options:

```python
from mloda.community.feature_groups.data_operations.aggregation.base import AggregationFeatureGroup
from mloda.provider import ApiDataFG
from mloda.user import Feature, Options, PluginCollector, PluginLoader, mloda

PluginLoader.all()

data = {
    "region": ["A", "A", "B", "B"],
    "category": ["X", "Y", "X", "X"],
    "value_int": [10, 5, 20, 30],
}

# Sum of value_int where category equals 'X', partitioned by region
feature = Feature(
    "value_int__sum_agg",
    Options(
        context={
            "partition_by": ["region"],
            "mask": ("category", "equal", "X"),
        }
    ),
)

result = mloda.run_all(
    [feature],
    compute_frameworks=["PandasDataFrame"],
    api_data={"MaskData": data},
    plugin_collector=PluginCollector.enabled_feature_groups({AggregationFeatureGroup, ApiDataFG}),
)
# Region A: the Y row is masked (5), leaving 10. Region B: 20 + 30 = 50.
assert sorted(result[0]["value_int__sum_agg"]) == [10, 50]
```

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

---

## Framework Behavior

The framework selection does not affect the mask spec format. The same `("column", "operator", value)` tuples work across all frameworks.

---

## Testing with MaskTestMixin

Compose `MaskTestMixin` (unit-level scenarios) with `DataOpsTestBase` and override its `mask_*` class methods, or `MaskIntegrationTestMixin` (pipeline-level, through `mloda.run_all`) with `DataOpsIntegrationTestBase` and override its `mask_integration_*` class methods; see [Adding a New Operation: Step 7](../data-operation-patterns/10-adding-new-operation.md#step-7-decide-if-it-needs-mask-support) and Step 6 for the framework test classes.

---

## Related

- [Filter Concepts](15-filter-concepts.md) - Filters remove rows; masks null out values
- [Options](11-options.md) - Passing mask specs via context options
- [Testing Guide](10-testing-guide.md) - Testing levels for feature groups
