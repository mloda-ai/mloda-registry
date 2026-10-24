# Key lookup (lookup)

Key lookup replaces each source value with the value its key maps to in a mapping, for example a product code with a category or a country code with a region.

**What**: `LookupFeatureGroup` (an `OperationFeatureGroup`, with one `Implementation` per framework) accepts feature names of the form `{col}__lookup`. It reads `lookup_mapping` and an optional `on_unmatched` policy from `Options(context=...)`, and emits a new `{col}__lookup` column.
**When**: You have a coded column and a key-to-value table, and want the translated value on every row.
**Why**: Unmatched keys, nulls and the output type behave the same on every backend.
**Where**: `mloda/community/feature_groups/data_operations/row_preserving/lookup/`.
**How**: The name carries only the op; the mapping and policy come from Options. Enable or scope the family by `LookupFeatureGroup`, and pin one backend with `Feature(..., implementation=PandasLookup)` (the class) or the qualified string `"module:Class"`.

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

data = {"product_code": ["A100", "B200", "C300"]}

features = [
    Feature(
        "product_code__lookup",
        Options(context={"lookup_mapping": {"A100": "tools", "B200": "toys"}, "on_unmatched": "null"}),
    ),
]
result = mloda.run_all(features, compute_frameworks=["PandasDataFrame"], api_data={"Src": data})
```

---

## Semantics

- **Row-preserving.** Same rows in the same order, one new column appended. The source column is left unchanged.
- **Key rule.** Keys are all `str` or all `int` (no null, bool or float), matched against the source column type. A mismatch, such as int keys on a string column, raises `ValueError` naming the column's dtype.
- **All-null rule.** A source column with no key type (null-typed, or an all-null column a backend stores without a type) yields an all-null output under every policy. A typed column is still checked. See the divergences for [DuckDB](known-divergences.md#duckdb-reads-a-null-typed-lookup-source-as-integer) (a null column becomes `INTEGER`) and [pandas](known-divergences.md#pandas-treats-a-typed-all-null-int-source-as-untyped) / [python dict](known-divergences.md#python-dict-treats-a-typed-all-null-lookup-source-as-untyped) (a typed all-null column is seen as untyped).
- **Value rule.** Values are non-null and all one type: `str`, `int`, `float` or `bool`. The output type follows: `STRING`, `INT64`, `DOUBLE` or `BOOLEAN`.
- **Null keys.** A null source value stays null under every policy and is never counted as unmatched.
- **Unmatched.** A non-null source value that is not a key. Policy `raise` (default) raises `LookupUnmatchedError`, `null` outputs null, `keep` passes the source value through.
- **`keep` needs key type == value type** (str to str, or int to int). Otherwise the generic operation raises `ValueError` when the feature is planned (match time), not only at compute; a hook's `keep` mismatch is reported when the feature is computed. The output type is then the value type.
- **Invalid mapping.** An empty mapping, mixed key types, bool, float or null keys, mixed or null value types, NaN values, or ints outside int64 are invalid. A generic feature with one does not match. A mapping from a plugin hook raises `ValueError` at compute time.
- **Pandas and nulls.** Pandas stores an int column with nulls as float64; see [the pandas entry](known-divergences.md#pandas-classes-float64-sources-as-int-keys).
- **Unsupported sources.** Ints outside int64 in the source (`uint64`, `UBIGINT`/`HUGEINT` in DuckDB, or Python/object ints) raise `ValueError` on every backend. Int keys outside the source width never match.

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

Use the hook instead of the option for large or external tables.

| Member | Meaning |
|---|---|
| `MAPPING_FROM_OPTION = False` | The subclass supplies its own mapping and matches by feature name only. It does not match through the config path. |
| `PREFIX_PATTERN` | Required, must differ from the base `r".*__lookup$"` (a subclass that keeps it raises `TypeError` at class definition). |
| `resolve_mapping(feature)` | Returns the mapping. Validated with the same rules as the option. |
| `resolve_unmatched_reasons(feature)` | Optional. Returns `{key: reason}` for the error message. |

```py
class NutsMapping:
    MAPPING_FROM_OPTION = False
    PREFIX_PATTERN = r".*__nuts2024$"

    @classmethod
    def resolve_mapping(cls, feature):
        return load_nuts_table()


class NutsLookup(NutsMapping, LookupFeatureGroup):
    pass
```

One subclass serves every framework: the shipped Implementations (`PyArrowLookup`, `PandasLookup`, `PolarsLazyLookup`, `DuckdbLookup`, `SqliteLookup`, `PythonDictLookup`) compute it. Subclass `LookupFeatureGroup`, not a framework class such as `PyArrowLookup`: those are Implementations, not FeatureGroups.

A hook rejects a `lookup_mapping` set on its own feature (`ValueError` at planning time). A `lookup_mapping` forwarded from a consumer's group options is ignored, so a generic `__lookup` can sit on top of a hook feature.

A hook over a generic lookup (`{col}__lookup__{hook}`) passes the group `lookup_mapping` to the inner generic feature, and a forwarded `on_unmatched` also configures the hook.

`resolve_mapping` also runs at planning time, because the default `return_data_type_rule` reads the value type from it. Cache an expensive load, or override `return_data_type_rule`. I/O errors in the hook propagate, per the [fail-fast contract](16-return-data-type-rule.md).

---

## Cross-framework notes

| Backend | Mechanism |
|---|---|
| PyArrow (reference) | `pc.index_in` on the keys, then `take` on the values; `keep` via `pc.if_else`. |
| Pandas | Exact index lookup on the keys (not `Series.map` or `isin`). Int output with nulls uses nullable `Int64`. |
| Polars (lazy) | `replace_strict` stays lazy. Policy `raise` evaluates the upstream plan once more with one grouped query. |
| DuckDB | The mapping is an Arrow relation, LEFT JOINed on the key with a row-number tag to restore order. Policy `raise` runs one grouped query. |
| SQLite | The mapping goes into a temp table through `executemany`, LEFT JOINed on the key with row order restored. |
| Python dict | Native dict lookup. |

SQL backends bind the mapping as data and never interpolate it into the query text.

---

## Related

- [Row-preserving contract](02-row-preserving-contract.md) - Output row count and order must match input.
- [`return_data_type_rule` failure handling](16-return-data-type-rule.md) - Why the hook's planning-time call must not hide errors.
