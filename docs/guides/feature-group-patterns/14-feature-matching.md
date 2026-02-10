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

### Why Unique Keys?

- **Semantic clarity**: `scaler_type` is clearer than generic `operation_type`
- **No collisions**: Different feature group types can have overlapping method names
- **Self-documenting**: The key name indicates which base class handles it
- **Follows mloda patterns**: Consistent with AggregatedFeatureGroup, ScalingFeatureGroup, etc.

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
