# calculate_feature

How to implement the core computation method of a FeatureGroup.

**What**: The method where actual feature computation happens.
**When**: Called by mloda after dependencies are resolved and data is available.
**Why**: Separates static class definition (matching, dependencies) from runtime computation.
**Where**: `calculate_feature(cls, data, features)` in every FeatureGroup.

A FeatureGroup class defines static behavior (matching, dependencies, framework rules). The `calculate_feature()` method receives runtime context: the actual data and a `FeatureSet` containing which features were requested, their options, and any filters. This separation allows one FeatureGroup class to handle many feature variants.

## Signature

```python
@classmethod
def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
```

| Parameter | Contains |
|-----------|----------|
| `data` | Input data (DataFrame, dict, etc.) with dependencies already computed |
| `features` | Runtime context: requested features, options, filters |

## FeatureSet Attributes

| Attribute | Type | Purpose |
|-----------|------|---------|
| `features` | `Set[Feature]` | All features to compute |
| `filters` | `Set[SingleFilter]` | Filters to apply (for data sources) |

## Common Pattern

```python
@classmethod
def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
    for feature in features.features:
        name = feature.name
        # Access options per feature
        threshold = feature.options.get("threshold") or 0.5
        data[name] = data["source"] > threshold
    return data
```

## Extracting the Operation Type

Chained feature groups (using `FeatureChainParserMixin`) often need to extract an operation type inside `calculate_feature`. The operation can come from the feature name string or from a config key in options. Use `_resolve_operation()` to handle both paths in one call:

```python
@classmethod
def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
    for feature in features.features:
        name = feature.name

        # Resolves from PREFIX_PATTERN match or options["imputation_method"]
        method = cls._resolve_operation(feature, "imputation_method")
        source = next(iter(feature.options.get_in_features())).name

        col = data[source]
        data[name] = col.fillna(col.mean() if method == "mean" else col.median())
    return data
```

`_resolve_operation(feature, config_key)` tries string-based parsing via `PREFIX_PATTERN` first, then falls back to `options.get(config_key)`. See [Chained Features](03-chained-features.md) for the full pattern.

---

## Related

- [Options](11-options.md) - Accessing feature options
- [Filter Concepts](15-filter-concepts.md) - Using filters in data sources
- [Chained Features](03-chained-features.md) - FeatureChainParserMixin and `_resolve_operation()`

## Full Documentation

See [FeatureGroup API](https://mloda-ai.github.io/mloda/in_depth/feature-group/) for detailed patterns.
