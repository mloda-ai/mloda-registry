# String Operations

Element-wise string transforms: uppercase, lowercase, trim, length, reverse. Row-preserving by construction.

**What**: `StringFeatureGroup` handles feature names of the form `{col}__{op}` where `op` is one of `upper`, `lower`, `trim`, `length`, `reverse`.
**When**: You need cleaned or derived text alongside the original column.
**Why**: These cover the overwhelmingly common cases. Each maps directly to a native function in every target framework, so there is no interpretation layer.
**Where**: `mloda/community/feature_groups/data_operations/string/`.
**How**: Name the feature; no context options are required.

---

## Operations

```python
# mloda/community/feature_groups/data_operations/string/base.py
STRING_OPS = {
    "upper":   "Convert string to uppercase",
    "lower":   "Convert string to lowercase",
    "trim":    "Strip leading and trailing whitespace",
    "length":  "Return the length of the string (integer)",
    "reverse": "Reverse the string",
}

PREFIX_PATTERN = r".+__(upper|lower|trim|length|reverse)$"
```

`length` returns an integer; the other four return strings.

---

## Usage

```python
from mloda.user import Feature, PluginLoader, mloda

PluginLoader.all()

features = [
    Feature("name__upper"),
    Feature("description__trim"),
    Feature("name__length"),
]

result = mloda.run_all(features, compute_frameworks={"PandasDataFrame"})
```

No `Options` needed. The feature name alone is enough.

---

## NULL handling

All five operations propagate NULL. A NULL input produces a NULL output; they never raise and never substitute an empty string.

---

## Framework differences

| Framework | Native mapping |
|---|---|
| PyArrow | `pc.utf8_upper`, `pc.utf8_lower`, `pc.utf8_trim_whitespace`, `pc.utf8_length`, `pc.utf8_reverse` |
| Pandas | `.str.upper()`, `.str.lower()`, `.str.strip()`, `.str.len()`, `.str[::-1]` (NULL-safe) |
| Polars lazy | `col.str.to_uppercase()`, `.to_lowercase()`, `.strip_chars()`, `.len_chars()`, `.reverse()` |
| DuckDB | `UPPER`, `LOWER`, `TRIM`, `LENGTH`, `REVERSE` |
| SQLite | `UPPER`, `LOWER`, `TRIM`, `LENGTH` only |

SQLite has no native `REVERSE`. The implementation refuses to match `__reverse` at resolution time rather than emulating it in SQL:

```python
# mloda/community/feature_groups/data_operations/string/sqlite_string.py
_SQLITE_STRING_EXPRS: dict[str, str] = {
    "upper":  "UPPER({col})",
    "lower":  "LOWER({col})",
    "trim":   "TRIM({col})",
    "length": "LENGTH({col})",
}

@classmethod
def _validate_string_match(cls, feature_name, operation_config, source_feature) -> bool:
    return operation_config in _SQLITE_STRING_EXPRS
```

The SQLite test class mirrors the restriction:

```python
class TestSqliteStringOps(SqliteTestMixin, StringTestBase):
    @classmethod
    def supported_ops(cls) -> set[str]:
        return {"upper", "lower", "trim", "length"}
```

If you request `name__reverse` with `compute_frameworks={"SqliteRelation"}`, the SQLite feature group will not match and the engine falls back to resolving the feature elsewhere (or errors). See [Supported ops](04-supported-ops.md) for how this pattern generalizes.

---

## Adding new string operations

String operations are intentionally narrow. If you need `regex_replace`, `split`, `concat`, etc., build them as a new feature group rather than extending `StringFeatureGroup`. A new feature group gives you room for the extra parameters (regex patterns, split delimiters, concat lists) that a bare suffix in the feature name cannot encode cleanly.

---

## Related

- [Supported ops per framework](04-supported-ops.md) - The mechanism SQLite uses to exclude `reverse`.
- [Adding a new data operation](10-adding-new-operation.md) - Build a separate feature group for richer string ops.
- [Feature naming](../feature-group-patterns/13-feature-naming.md) - General rules for the `{col}__{op}` convention.
