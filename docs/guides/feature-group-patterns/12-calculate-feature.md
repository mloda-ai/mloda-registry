# calculate_feature

How to implement the core computation method of a FeatureGroup.

**What**: The method where actual feature computation happens.
**When**: Called by mloda after dependencies are resolved and data is available.
**Why**: Separates static class definition (matching, dependencies) from runtime computation.
**Where**: `calculate_feature(cls, data, features)` in every FeatureGroup.

## Signature

```py
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

```py
@classmethod
def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
    for feature in features.features:
        name = feature.name
        # Access options per feature
        threshold = feature.options.get("threshold") or 0.5
        data[name] = data["source"] > threshold
    return data
```

### A Feature Requested Directly and as an Input

A feature requested directly and also as another requested feature's input reaches your group once: mloda merges the two copies at intake, so no per-group dedupe is needed. The exception is a consumer that pushes a context key down with `propagate_context_keys`: the copies then differ but would share one frame, so resolution rejects it up front with a `ValueError` naming the feature and the key. Remedies: give every request the same value, push the key with `cascade_context_keys` so the inputs split too, or make it a group option.

## Which Features Share a Call

One call receives the requested features of your group that agree on:

- group options (context does not split, except a key a feature received from its consumer through `inherit_context_keys`, `propagate_context_keys` or `cascade_context_keys`, which splits all of the group's features by value)
- the compute framework they run on
- the data source the group matched, and for API input data the API source
- the variant of a source class their inputs read from

Features that differ only in data type share a call when they carry the same filters, unless the group has an artifact or two of them share a name. A feature that needs another feature of the same group runs in a later call. A cache scoped to one call therefore shares work only within that call: to compute siblings once, make them [declared parts](05-multi-output-features.md#declaring-output-parts) of one feature (users then request `scan~label`; parts requested with differing data types still split, since they share the base name), or cache beyond the call, keyed by the inputs. `calculate_feature` may run concurrently in one process (`THREADING` steps, or `run_all` or `session.run` calls from several threads, as in a threading server), so replace such a cache whole (copy-on-write) or guard it with a lock.

```python
from typing import Any

from mloda.provider import DataCreatorFeatureGroup, FeatureSet
from mloda.user import Feature, PluginCollector, PluginLoader, mloda

PluginLoader.all()

calls: list[set[str]] = []


class Guide12Scan(DataCreatorFeatureGroup):
    """Serves a label and two counts, recording the features each call receives."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"guide12_label", "guide12_rows", "guide12_cols"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        names = {str(f.name) for f in features.features}
        calls.append(names)
        values = {"guide12_label": ["x"], "guide12_rows": [3], "guide12_cols": [2]}
        return {name: values[name] for name in names}


mloda.run_all(
    [
        Feature.str_of("guide12_label"),
        Feature.int64_of("guide12_rows"),
        Feature.int64_of("guide12_cols", {"guide12_scope": "b"}),
    ],
    compute_frameworks=["PandasDataFrame"],
    plugin_collector=PluginCollector.enabled_feature_groups({Guide12Scan}),
)
assert sorted(sorted(c) for c in calls) == [["guide12_cols"], ["guide12_label", "guide12_rows"]]
```

## Return Contract

The framework's `transform()` accepts two shapes, and which one you return decides whether you **replace** the frame or **append** a column:

| Return | Meaning |
|--------|---------|
| A whole frame | Initial data. Becomes the frame. Returned by root features, by `replaces_rows()` groups, and by a group that returns `data` with its column added. |
| A single column | Added data. Appended to the existing frame under the requested name. Only valid when the FeatureSet holds exactly one feature. |

| Framework | Whole frame | Single column |
|-----------|-------------|---------------|
| `PythonDictFramework` | Columnar `dict[str, list[Any]]` | None: see below |
| `PandasDataFrame` | `pd.DataFrame` | `pd.Series` |
| `PolarsDataFrame` / `PolarsLazyDataFrame` | `pl.DataFrame` / `pl.LazyFrame` | `pl.Series` |
| `PyArrowTable` | `pa.Table` | `pa.Array` / `pa.ChunkedArray` |
| `DuckDBFramework` | `duckdb.DuckDBPyRelation` | Any iterable of values |
| `SqliteFramework` | `SqliteRelation` | Any iterable of values |
| `SparkFramework` | Spark `DataFrame` | Any iterable of values |

Every framework above also accepts a columnar `dict[str, list]` as the whole-frame form, not just `PythonDictFramework`.

### PythonDictFramework Is Columnar

PythonDictFramework's native structure is columnar `dict[str, list[Any]]`: one key per column, all value-lists the same length.

```py
@classmethod
def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
    return {"doc_id": [1, 2], "score": [0.9, 0.4]}  # root: this becomes the frame
```

It has **no single-column form**. A bare `list` is read as a list of row dicts, not as one column's values. Because the returned dict is already the native type, `transform()` is skipped and the dict *becomes* the frame, so a derived feature that returns only its own column silently drops every input column:

```py
@classmethod
def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
    # WRONG: replaces the frame. A downstream group needing "src"
    # then fails with ValueError: Feature '...' failed with a KeyError: 'src'
    return {"doubled": [v * 2 for v in data["src"]]}

    # RIGHT: add the column, return the whole dict (the analogue of pandas data[name] = ...)
    data["doubled"] = [v * 2 for v in data["src"]]
    return data
```

A row-oriented `list[dict]` is still accepted, but it is not the native shape: `transform()` pivots it with `rows_to_columnar()` on every calculation, and that pivot requires **homogeneous keys** across rows. A row missing a key raises `ValueError: Inconsistent row keys at index i`. Build the columnar dict directly, or run rows through `homogenize_rows()` first.

### Returning Other Rows: replaces_rows()

Override `replaces_rows()` to return True when the group returns other rows than it receives (an aggregation, a resample or re-basing, a sorted, filtered or deduplicated table). Each step then writes a frame of its own while sibling steps keep reading the unchanged input, so return a new whole frame holding the output and the key column a consumer joins on (input columns it does not return are not on its frame; consumers reach them through the Link described below), and never add columns to `data` or otherwise mutate it. On `PythonDictFramework` that is a fresh dict, the shape the "WRONG" example above warns about for row-preserving groups. Core handles the flag, not the compute framework, so it works on every built-in framework (and any whose `transform()` accepts a whole frame): return that framework's whole-frame form from the table above. The aggregation group in [`aggregation/base.py`](https://github.com/mloda-ai/mloda-registry/blob/main/mloda/community/feature_groups/data_operations/aggregation/base.py) is a real example.

```py
@classmethod
def replaces_rows(cls) -> bool:
    return True
```

A consumer reading the output together with an unchanged input column, or with another `replaces_rows()` output, needs a `Link` (see [Joining a replaces_rows() Output](08-links-joins.md#joining-a-replaces_rows-output)). Without the flag such a consumer either fails only at run time with `MissingInputFeatureError`, which does not name the cause, or, when the output keeps the row count, silently reads misaligned rows. For a runnable check see [Refusing a Consumer of a Row-Replacing Output](10-testing-guide.md#refusing-a-consumer-of-a-row-replacing-output) and mloda's [result contract](https://github.com/mloda-ai/mloda/blob/0.20.0/docs/docs/chapter1/feature-groups.md#5-the-result-contract).

## Reporting Failed Rows

When a row got no value (a lookup miss, a parse failure), call `report_failed_rows(column, rows, reason)` from `mloda.provider` instead of returning a silent null. `reason` is a short code matching `[A-Za-z0-9_.-]{1,64}`, never row data (anything else raises `ValueError`). `rows` are positions in the group's own output.

A direct call outside a run is a no-op. In every run, streams included, the entries reach `HookContext.failed_rows` and the extenders; `run()` and `run_all()` also return them in `RunResult.failed_rows`. The extenders turn them into counts and reason codes: `mloda.failed_rows.*` span attributes, the `mloda.step.failed_rows` counter, the `mlodaFailedRows` lineage facet and the audit `failed_rows` key. Positions never leave the process. See the core [result contract](https://github.com/mloda-ai/mloda/blob/0.20.0/docs/docs/chapter1/feature-groups.md#5-the-result-contract).

```python
from typing import Any

from mloda.provider import DataCreatorFeatureGroup, FailedRows, FeatureSet, report_failed_rows
from mloda.steward import collect_failed_rows
from mloda.user import Feature, PluginCollector, PluginLoader, mloda

PluginLoader.all()


class Guide12Prices(DataCreatorFeatureGroup):
    """Serves prices, reporting the unparsable ones."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"guide12_price"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raw = ["1.5", "n/a", "2.0"]
        bad = [i for i, v in enumerate(raw) if not v.replace(".", "", 1).isdigit()]
        report_failed_rows("guide12_price", bad, "parse_error")
        return {"guide12_price": [None if i in bad else float(v) for i, v in enumerate(raw)]}


result = mloda.run_all(
    [Feature.double_of("guide12_price")],
    compute_frameworks=["PythonDictFramework"],
    plugin_collector=PluginCollector.enabled_feature_groups({Guide12Prices}),
)
assert result.failed_rows == (FailedRows(Guide12Prices.get_class_name(), "guide12_price", (1,), "parse_error"),)

with collect_failed_rows("Guide12Prices") as collector:
    Guide12Prices.calculate_feature(None, FeatureSet())
assert collector.entries == [FailedRows("Guide12Prices", "guide12_price", (1,), "parse_error")]
```

## Empty Results

Zero rows is a valid result. Zero columns is not: a FeatureGroup must always return its schema, so a result with no columns raises after calculation.

```python
from mloda.provider import EmptyResultError
```

```text
EmptyResultError: Result carries no schema (no columns): <FG>. A feature must
return a schema; zero rows is a valid result, zero columns is not.
```

For `PythonDictFramework` this means returning `[]` or `{}` to mean "nothing found" is a bug, because both normalize to the schema-less `{}`. Name the columns instead:

```py
if not hits:
    return {"doc_id": [], "score": []}  # zero rows, schema intact
```

A row-oriented `list[dict]` return cannot express "zero rows with schema" at all, since an empty list carries no keys. Any FeatureGroup with a reachable empty path (empty source directory, query with no hits, a [filter](15-filter-concepts.md) that empties the result) must therefore be columnar.

`EmptyResultError` is raised when the schema-less group's feature was requested from the API. An intermediate group that returns no schema fails later instead, as a missing-column error in whichever group consumed it. Return your columns either way.

The check runs during execution, not inside `calculate_feature()`, so a unit test calling `calculate_feature()` directly will not catch a violation. See [Testing Guide](10-testing-guide.md).

## Columnar Helpers

`mloda.user.python_dict` exports helpers for the `PythonDictFramework` shape. Use them instead of hand-rolling conversions:

| Helper | Purpose |
|--------|---------|
| `rows_to_columnar(rows)` | Pivot `list[dict]` to columnar; requires homogeneous keys |
| `columnar_to_rows(data)` | Pivot columnar to `list[dict]`, e.g. to feed a row-wise library |
| `is_columnar(data)` | True iff `data` is a dict of equal-length lists |
| `homogenize_rows(rows)` | Fill missing keys with `None` so rows pivot cleanly |

```python
from mloda.user.python_dict import columnar_to_rows, homogenize_rows, is_columnar, rows_to_columnar
```

## Extracting the Operation Type

Chained groups read a value that can come from the name or from options with `cls._resolve_operation(feature, key)`; see [Chained Features](03-chained-features.md) for the lookup rules.

Framework-specific column access belongs in one `Implementation` per framework behind the operation group. See [Framework-Specific](09-framework-specific.md#complete-example).

---

## Related

- [Options](11-options.md) - Accessing feature options
- [Filter Concepts](15-filter-concepts.md) - Using filters in data sources
- [Chained Features](03-chained-features.md) - FeatureChainParserMixin and `_resolve_operation()`

## Full Documentation

See [FeatureGroup API](https://mloda-ai.github.io/mloda/in_depth/feature-group/) for detailed patterns.
