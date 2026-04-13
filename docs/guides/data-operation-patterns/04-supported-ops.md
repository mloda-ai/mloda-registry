# Supported Ops Per Framework

Not every framework can express every operation. SQLite has no native REVERSE; some frameworks lack a native NTILE. The `supported_ops()` pattern lets a framework test class declare what it can handle so the shared tests skip the rest cleanly.

**What**: A class method on the test class that returns the set of op names the framework implements.
**When**: Your framework implementation does not cover every op listed in the operation's base class.
**Why**: Without it, inherited tests for unsupported ops would fail with "operation not matched" errors. With it, they skip with a clear message.
**Where**: Defined on the operation's test base (e.g. `BinningTestBase.supported_ops`) and overridden on the concrete framework test class.
**How**: Override `supported_ops()` to return the subset. `_skip_if_unsupported(op)` in `DataOpsTestBase` handles the skip logic.

---

## How `_skip_if_unsupported` finds the set

`DataOpsTestBase._skip_if_unsupported` probes four class methods in order and uses whichever is defined:

```python
for attr in ("supported_agg_types", "supported_ops", "supported_offset_types", "supported_rank_types"):
    method = getattr(self, attr, None)
    if method is not None:
        if op not in method():
            pytest.skip(f"{op} not supported by this framework")
        return
```

The multiple method names exist because different operation categories use different vocabulary: aggregation has "agg_types", binning has "ops", offset has "offset_types", rank has "rank_types". The semantics are the same.

---

## Example: SQLite excludes `upper`, `lower`, and `reverse`

SQLite has no native `REVERSE`, and its native `UPPER`/`LOWER` are ASCII-only and would diverge from the PyArrow reference on non-ASCII input (e.g. `UPPER('héllo')` returns `'HéLLO'` instead of `'HÉLLO'`). Rather than silently produce divergent results or emulate in SQL, the framework implementation refuses to match all three at selection time:

```python
# mloda/community/feature_groups/data_operations/string/sqlite_string.py
_SQLITE_STRING_EXPRS: dict[str, str] = {
    "trim":   "TRIM({col})",
    "length": "LENGTH({col})",
}

@classmethod
def _validate_string_match(cls, feature_name, operation_config, source_feature) -> bool:
    return operation_config in _SQLITE_STRING_EXPRS
```

The test class mirrors that decision:

```python
# mloda/community/feature_groups/data_operations/string/tests/test_sqlite.py
class TestSqliteStringOps(SqliteTestMixin, StringTestBase):
    @classmethod
    def supported_ops(cls) -> set[str]:
        return {"trim", "length"}
```

With this override, the inherited `test_upper_*`, `test_lower_*`, and `test_reverse_*` methods skip with "not supported by this framework" instead of failing.

---

## Two places, two responsibilities

The test-side `supported_ops()` is only for test skipping. It does **not** by itself change what feature names the framework accepts at runtime. The two declarations are independent:

| Declaration | Lives in | Purpose |
|---|---|---|
| `_validate_string_match` / `match_feature_group_criteria` override | `framework_{op}.py` | Refuses the feature name at resolution time so the mloda engine routes elsewhere (or errors). |
| `supported_ops()` (or `supported_agg_types`, etc.) | `tests/test_{framework}.py` | Makes the inherited test suite skip the ones the framework does not claim. |

Keep them consistent. If your implementation silently drops an op at runtime but the test class still claims to support it, the suite will fail. If the test class over-restricts, you lose coverage for something you actually implemented.

---

## Adding a new framework for an existing op

When you add, say, a Polars-lazy implementation for an operation, copy the test template:

```python
from mloda.testing.feature_groups.data_operations.mixins.polars_lazy import PolarsLazyTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.rank.rank import RankTestBase

class TestPolarsLazyRank(PolarsLazyTestMixin, RankTestBase):
    @classmethod
    def implementation_class(cls):
        return PolarsLazyRank

    @classmethod
    def supported_rank_types(cls) -> set[str]:
        return {"row_number", "rank", "dense_rank", "percent_rank"}
        # ntile_N omitted if Polars version lacks a clean NTILE equivalent
```

Start by returning the full set the operation supports. Shrink it only as concrete limitations appear.

---

## Related

- [Framework support matrix](framework-support-matrix.md) - The capability matrix this mechanism produces, rendered as operation x framework tables.
- [Reference implementation pattern](03-reference-implementation.md) - How framework implementations are compared against PyArrow.
- [String operations](09-string-operations.md) - The most common place `supported_ops()` comes up.
- [Adding a new data operation](10-adding-new-operation.md) - How to define the method on a new operation's test base.
