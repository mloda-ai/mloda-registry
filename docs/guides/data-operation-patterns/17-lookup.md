# Key lookup (lookup)

Key lookup replaces each source value with the value its key maps to in a mapping, for example a product code with a category or a country code with a region.

**What**: `LookupFeatureGroup` accepts feature names of the form `{col}__lookup`. It reads `lookup_mapping` and an optional `on_unmatched` policy from `Options(context=...)`, and emits a new `{col}__lookup` column.
**When**: You have a coded column and a key-to-value table, and want the translated value on every row.
**Why**: Unmatched keys, nulls and the output type behave the same on every backend.
**Where**: `mloda/community/feature_groups/data_operations/row_preserving/lookup/`.
**How**: The name carries only the op; the mapping and policy come from Options.

---

## Pattern

```text
product_code__lookup
```

| Option | Required | Meaning |
|---|---|---|
| `lookup_mapping` | yes (generic op) | Non-empty mapping from source keys to output values. |
| `on_unmatched` | no (default `raise`) | `raise`, `null` or `keep`: what to do with a non-null source value that is not a key. |
| `in_features` | config form only | The source column (when not using the `{col}__lookup` string form). |

```python
from mloda.user import Feature, Options, PluginLoader, mloda

PluginLoader.all()

features = [
    Feature(
        "product_code__lookup",
        Options(context={"lookup_mapping": {"A100": "tools", "B200": "toys"}, "on_unmatched": "null"}),
    ),
]
result = mloda.run_all(features, compute_frameworks=["PandasDataFrame"])
```

---

## Semantics

- **Row-preserving.** Same rows in the same order, one new column appended. The source column is left unchanged.
- **Key rule.** Keys are all `str` or all `int` (no null, bool or float), matched against the source column type. A mismatch, such as int keys on a string column, raises `ValueError` before any engine call.
- **Value rule.** Values are non-null and all one type: `str`, `int`, `float` or `bool`. The output type follows: `STRING`, `INT64`, `DOUBLE` or `BOOLEAN`.
- **Null keys.** A null source value stays null under every policy and is never counted as unmatched.
- **Unmatched.** A non-null source value that is not a key. Policy `raise` (default) raises `LookupUnmatchedError`, `null` outputs null, `keep` passes the source value through.
- **`keep` needs key type == value type** (str to str, or int to int). Otherwise `ValueError` is raised before dispatch. The output type is then the value type.
- **Invalid mapping.** An empty mapping, mixed key types or mixed value types does not match as a generic feature. A mapping from a plugin hook raises `ValueError` at compute time.
- **Pandas and nulls.** Pandas stores an int column with nulls as float64. A float64 source whose non-null values are all integral counts as int keys. A float64 source with fractional values is rejected like any other non-int, non-str source.

## LookupUnmatchedError

A `ValueError` subclass raised by policy `raise`. Attributes:

| Attribute | Meaning |
|---|---|
| `feature_name` | The output feature name. |
| `unmatched_rows` | Number of rows with an unmatched key. |
| `distinct_keys` | Number of distinct unmatched keys. |
| `sample` | Tuple of up to 10 unmatched keys, sorted. |
| `reasons` | Reasons from `resolve_unmatched_reasons`, restricted to the sample. |

```text
product_code__lookup: 3 rows, 2 distinct keys not in the mapping. Sample: ['C300', 'D400']
```

A key with a reason renders as `'C300' (reason text)`.

---

## Plugin hook subclass

A dict option suits small maps. Large or external tables (a code list, a database table) go through the hook instead of the options.

| Member | Meaning |
|---|---|
| `MAPPING_FROM_OPTION = False` | The subclass supplies its own mapping and matches by feature name only. It does not match through the config path. |
| `PREFIX_PATTERN` | Required, must differ from the base `r".*__lookup$"` (a subclass that keeps it raises `TypeError` at class definition). |
| `resolve_mapping(feature)` | Returns the mapping. Validated with the same rules as the option. |
| `resolve_unmatched_reasons(feature)` | Optional. Returns `{key: reason}` for the error message. |

```python
class NutsMapping:
    MAPPING_FROM_OPTION = False
    PREFIX_PATTERN = r".*__nuts2024$"

    @classmethod
    def resolve_mapping(cls, feature):
        return load_nuts_table()


class PyArrowNutsLookup(NutsMapping, PyArrowLookup):
    pass
```

Add one such subclass per framework (`PandasLookup`, `PolarsLazyLookup`, `DuckdbLookup`, `SqliteLookup`, `PythonDictLookup`) to reuse the mapping class.

`resolve_mapping` also runs at planning time, because the default `return_data_type_rule` reads the value type from the mapping. Cache an expensive load, or override `return_data_type_rule` to return the type without resolving. I/O errors in the hook propagate, per the [fail-fast contract](16-return-data-type-rule.md).

---

## Cross-framework notes

| Backend | Mechanism |
|---|---|
| PyArrow (reference) | `pc.index_in` on the keys, then `take` on the values; `keep` via `pc.if_else`. |
| Pandas | `Series.map` for values, `isin` to flag unmatched keys. Int output with nulls uses nullable `Int64`. |
| Polars (lazy) | `replace_strict` stays lazy. Policy `raise` collects the unmatched keys eagerly. |
| DuckDB | The mapping is an Arrow relation, LEFT JOINed on the key with a row-number tag to restore order. Policy `raise` evaluates the unmatched keys eagerly. |
| SQLite | The mapping goes into a temp table through `executemany`, LEFT JOINed on the key with row order restored. |
| Python dict | Native dict lookup. |

SQL backends bind the mapping as data and never interpolate it into the query text.

---

## Related

- [Row-preserving contract](02-row-preserving-contract.md) - Output row count and order must match input.
- [`return_data_type_rule` failure handling](16-return-data-type-rule.md) - Why the hook's planning-time call must not hide errors.
