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

`features.filters` is always available, regardless of how the FeatureGroup's `final_filters()` is configured. Data sources typically read filters inline (e.g. building a WHERE clause), then return `False` from `final_filters()` to skip post-calculation row elimination. See [Filter Concepts](15-filter-concepts.md) for the full set of patterns.

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

## Return Contract

Return the native data structure of the compute framework you run on:

| Framework | Return |
|-----------|--------|
| `PythonDictFramework` | **Columnar** `dict[str, list[Any]]`: one key per column, all value-lists the same length |
| `PandasDataFrame` | `pd.DataFrame` |
| `PolarsDataFrame` / `PolarsLazyDataFrame` | `pl.DataFrame` / `pl.LazyFrame` |
| `PyArrowTable` | `pa.Table` |
| `DuckDBFramework` / `SqliteFramework` | Relation |
| `SparkFramework` | Spark `DataFrame` |

`PythonDictFramework` is columnar since mloda 0.9.0:

```python
@classmethod
def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
    return {"doc_id": [1, 2], "score": [0.9, 0.4]}  # native
```

A row-oriented `list[dict]` is still accepted, but it is not the native shape: `transform()` pivots it with `rows_to_columnar()` on every calculation, and that pivot requires **homogeneous keys** across rows. A row missing a key raises `ValueError: Inconsistent row keys at index i`. Build the columnar dict directly, or run rows through `homogenize_rows()` first.

## Empty Results

Zero rows is a valid result. Zero columns is not: a FeatureGroup must always return its schema, so a result with no columns raises after calculation.

```
EmptyResultError: Result carries no schema (no columns): <FG>. A feature must
return a schema; zero rows is a valid result, zero columns is not.
```

For `PythonDictFramework` this means returning `[]` or `{}` to mean "nothing found" is a bug, because both normalize to the schema-less `{}`. Name the columns instead:

```python
if not hits:
    return {"doc_id": [], "score": []}  # zero rows, schema intact
```

A row-oriented `list[dict]` return cannot express "zero rows with schema" at all, since an empty list carries no keys. Any FeatureGroup with a reachable empty path (empty source directory, query with no hits, a [filter](15-filter-concepts.md) that empties the result) must therefore be columnar.

The contract is only enforced through `mloda.run_all()`, so a unit test calling `calculate_feature()` directly will not catch a violation. See [Testing Guide](10-testing-guide.md).

## Columnar Helpers

`mloda.user` exports helpers for the `PythonDictFramework` shape. Use them instead of hand-rolling conversions:

| Helper | Purpose |
|--------|---------|
| `rows_to_columnar(rows)` | Pivot `list[dict]` to columnar; requires homogeneous keys |
| `columnar_to_rows(data)` | Pivot columnar to `list[dict]`, e.g. to feed a row-wise library |
| `is_columnar(data)` | True iff `data` is a dict of equal-length lists |
| `homogenize_rows(rows)` | Fill missing keys with `None` so rows pivot cleanly |

```python
from mloda.user import columnar_to_rows, homogenize_rows, is_columnar, rows_to_columnar
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
