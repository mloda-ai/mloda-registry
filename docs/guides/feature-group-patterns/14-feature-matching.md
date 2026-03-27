# Feature Matching

How mloda determines which FeatureGroup handles a requested feature name.

**What**: The process of finding which FeatureGroup handles a requested feature name.
**When**: Every feature request triggers matching to find the responsible FeatureGroup.
**Why**: Exactly one FeatureGroup must handle each feature name; matching resolves this.
**Where**: `match_feature_group_criteria()` method, checked against all registered FeatureGroups.

## How It Works

When you request a feature (e.g., `Feature.not_typed("price__scaled")`), mloda checks each FeatureGroup's `match_feature_group_criteria()` method. Exactly one must return `True`.

---

## Default Matching Priority

The default `match_feature_group_criteria()` checks in order:

```
1. Input data match     → Root features with matching input_data()
2. Data access match    → ConnectionMatcherMixin with data connection
3. Exact class name     → "MyFeature" == class MyFeature
4. Prefix match         → "MyFeature_x".startswith("MyFeature_")
5. Explicit names       → name in feature_names_supported()
```

First `True` wins. If FeatureChainParserMixin is used, pattern matching is also applied.

---

## Custom Matching Override

Override `match_feature_group_criteria()` for custom logic:

```python
@classmethod
def match_feature_group_criteria(
    cls,
    feature_name: str,
    options: Options,
    data_access_collection: Optional[DataAccessCollection] = None,
) -> bool:
    return feature_name.endswith("_score")
```

**Note:** This only controls MATCHING. It doesn't define discoverable names - users must know to request matching names.

---

## Discriminator Keys for Configuration-Based Matching

When multiple method variants exist for a feature group type (e.g., different scaling algorithms), use a **unique discriminator key** to distinguish them. The `FeatureChainParserMixin` handles matching automatically via `PREFIX_PATTERN` and `PROPERTY_MAPPING`.

### Existing mloda Examples

| Feature Group | Discriminator Key | PREFIX_PATTERN |
|--------------|------------------|----------------|
| `AggregatedFeatureGroup` | `aggregation_type` | `r".*__([\w]+)_aggr$"` |
| `ScalingFeatureGroup` | `scaler_type` | `r".*__(standard|minmax|robust|normalizer)_scaled$"` |
| `EncodingFeatureGroup` | `encoder_type` | `r".*__(onehot|label|ordinal)_encoded(~\d+)?$"` |

### Pattern

The base class defines `PREFIX_PATTERN` and `PROPERTY_MAPPING`. The mixin's `match_feature_group_criteria()` handles both string-based and config-based matching automatically:

```python
class BaseMyTransform(FeatureChainParserMixin, FeatureGroup):
    # String-based matching: extracts method from feature name
    PREFIX_PATTERN = r".*__(algo_a|algo_b)_transformed$"

    # Discriminator key for config-based matching
    MY_METHOD = "my_method"
    MY_METHODS = {
        "algo_a": "Algorithm A description",
        "algo_b": "Algorithm B description",
    }

    PROPERTY_MAPPING = {
        MY_METHOD: {
            **MY_METHODS,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature",
            DefaultOptionKeys.context: True,
        },
    }

    # No need to override match_feature_group_criteria() - mixin handles it
```

**Usage** - both approaches route to the same FeatureGroup:

```python
# String-based: method extracted from name
Feature("input__algo_b_transformed")

# Config-based: method specified in options
Feature("my_output", Options(context={"my_method": "algo_b", "in_features": "input"}))
```

**Note:** Subclasses typically only override `compute_framework_rule()` to specify which framework they support (Pandas, Polars, etc.), not matching logic.

### Extracting the Discriminator at Compute Time

At matching time, the mixin resolves string-based and config-based features automatically. At compute time (inside `calculate_feature`), use `_resolve_operation()` to extract the discriminator value without calling `FeatureChainParser` directly:

```python
@classmethod
def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
    for feature in features.features:
        # Resolves from PREFIX_PATTERN or options["my_method"]
        method = cls._resolve_operation(feature, cls.MY_METHOD)
        source = next(iter(feature.options.get_in_features())).get_name()
        data[feature.get_name()] = apply_transform(data[source], method)
    return data
```

### Why Unique Keys?

- **Semantic clarity**: `scaler_type` is clearer than generic `operation_type`
- **No collisions**: Different feature group types can have overlapping method names
- **Self-documenting**: The key name indicates which base class handles it
- **Follows mloda patterns**: Consistent with AggregatedFeatureGroup, ScalingFeatureGroup, etc.

---

## Conditional Requirements with `required_when`

Some PROPERTY_MAPPING entries only become required under certain conditions. For example, an `order_by` column is only needed when the aggregation type is `first` or `last`, but not for `sum` or `avg`.

Use `DefaultOptionKeys.required_when` to attach a predicate callable to a mapping entry. The predicate receives an effective `Options` object and returns `True` when the option is required. When the predicate returns `True` and the option value is absent, `match_feature_group_criteria` returns `False`. When the predicate returns `False`, the option is treated as optional.

```python
from mloda.core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

_ORDER_DEPENDENT = {"first", "last"}

def _needs_order_by(options: Options) -> bool:
    """order_by is required when aggregation_type is first or last."""
    return options.get("aggregation_type") in _ORDER_DEPENDENT

PROPERTY_MAPPING = {
    "aggregation_type": {
        "sum": "Sum", "avg": "Average", "first": "First", "last": "Last",
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: True,
    },
    "order_by": {
        "explanation": "Column to order by within each partition",
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: False,
        DefaultOptionKeys.required_when: _needs_order_by,
    },
}
```

### How It Works

1. When `match_feature_group_criteria` runs, it evaluates each `required_when` predicate against the effective options.
2. For string-based features, the operation value parsed from the feature name (e.g., `first` from `source__first_windowed`) is merged into effective options before predicate evaluation. This means predicates see values from both the feature name and explicit options.
3. Entries with `required_when` are treated as optional by the base parser. The predicate decides at runtime whether to require them.

### Before and After

Without `required_when`, you must override `match_feature_group_criteria` to manually check conditions:

```python
# Before: manual override with boilerplate
@classmethod
def match_feature_group_criteria(cls, feature_name, options, _dac=None):
    if not super().match_feature_group_criteria(feature_name, options, _dac):
        return False
    agg_type = cls._resolve_agg_type(feature_name, options)
    if agg_type in {"first", "last"}:
        order_by = options.get("order_by")
        if not isinstance(order_by, str):
            return False
    return True
```

With `required_when`, the mixin handles this automatically:

```python
# After: declarative, no override needed
PROPERTY_MAPPING = {
    "aggregation_type": { ... },
    "order_by": {
        "explanation": "Column to order by",
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: False,
        DefaultOptionKeys.required_when: _needs_order_by,
    },
}
# match_feature_group_criteria inherited from mixin - no override required
```

See [property-mapping.md](https://mloda-ai.github.io/mloda/in_depth/property-mapping/) in the core docs for the full reference.

---

## Custom Validation Functions

Use `DefaultOptionKeys.validation_function` to validate option values with a callable instead of checking membership against a fixed set of allowed values. This is useful when valid values cannot be enumerated (numeric ranges, dynamic lookups, type checks).

When `strict_validation` is `True` and a `validation_function` is present, the function is called instead of checking whether the value exists in the mapping dict. The function receives the option value and must return `True` if valid.

```python
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

PROPERTY_MAPPING = {
    "window_size": {
        "explanation": "Number of rows in the rolling window",
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: True,
        DefaultOptionKeys.validation_function: lambda x: isinstance(x, int) and x > 0,
    },
    "threshold": {
        "explanation": "Cutoff value for filtering",
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: True,
        DefaultOptionKeys.validation_function: lambda x: isinstance(x, (int, float)) and 0.0 <= x <= 1.0,
    },
}
```

### When to Use Each Approach

| Approach | Use When | Example |
|----------|----------|---------|
| Enumerated values | Fixed set of valid values | `"sum"`, `"avg"`, `"min"` |
| `validation_function` | Open-ended or numeric values | positive integers, float ranges |
| No strict validation | Any value is acceptable | free-text, dynamic feature names |

### Combining with `required_when`

Both features can be used together on the same mapping entry:

```python
PROPERTY_MAPPING = {
    "aggregation_type": {
        "sum": "Sum", "avg": "Average", "first": "First",
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: True,
    },
    "order_by": {
        "explanation": "Column to order by",
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: False,
        DefaultOptionKeys.required_when: _needs_order_by,
    },
    "window_size": {
        "explanation": "Rolling window size (positive integer)",
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: True,
        DefaultOptionKeys.validation_function: lambda x: isinstance(x, int) and x > 0,
        DefaultOptionKeys.required_when: lambda opts: opts.get("aggregation_type") == "sum",
    },
}
```

---

## Matching vs Naming

| Concept | Method | Purpose |
|---------|--------|---------|
| **Naming** | `feature_names_supported()`, class name | Define what names exist (discoverable) |
| **Matching** | `match_feature_group_criteria()` | Check if a requested name is handled |

See [Feature Naming](13-feature-naming.md) for defining names.

---

## Debugging Matching

Use `resolve_feature()` to verify your FeatureGroup matches correctly:

```python
from mloda.user import PluginLoader
from mloda.steward import resolve_feature

PluginLoader.all()

# Verify your pattern matches
result = resolve_feature("price__standard_scaled")
print(f"Resolved to: {result.feature_group.__name__}")
print(f"Candidates: {[fg.__name__ for fg in result.candidates]}")

# Debug a failing match
result = resolve_feature("my_custom_feature")
if result.error:
    print(f"Failed: {result.error}")
```

This is especially useful when:

- Your custom `match_feature_group_criteria()` isn't matching as expected
- Multiple FeatureGroups compete for the same pattern
- You want to verify subclass filtering works correctly

See [Discover Plugins](../02-discover-plugins.md#debugging-feature-resolution) for the full API reference.
