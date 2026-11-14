# Supported Ops Per Framework

Not every framework can express every operation. SQLite has no native REVERSE; some frameworks lack a native NTILE. A framework declares what it can handle on its Implementation, and the shared tests skip the rest cleanly.

**What**: The Implementation's `supported` names the subtypes the framework implements; the family test base reads it through `declared_supported`.
**When**: Your framework implementation does not cover every op listed in the operation's group.
**Why**: Without it, inherited tests for unsupported ops would fail with "operation not matched" errors. With it, they skip with a clear message.
**Where**: `supported` on the Implementation in `framework_{op}.py`. The test base's `supported_ops()` (or `supported_agg_types()`, etc.) defaults to `DataOpsTestBase.declared_supported(...)`, which returns the Implementation's `supported`, or the group's `subtype_universe()` when it is unrestricted.
**How**: Declare `supported` on the Implementation. `_skip_if_unsupported(op)` in `DataOpsTestBase` skips the inherited tests for the rest. The hooks stay overridable, but shipped test classes do not need to. Frame aggregate time units are read the same way from the Implementation's `SUPPORTED_TIME_UNITS`. A restriction expressed only in `accepts` is not read by the suite; `tests/test_twin_catalog_consistency.py` fails when the suite and the catalog's probe disagree.

---

## How `_skip_if_unsupported` finds the set

`DataOpsTestBase._skip_if_unsupported` probes four class methods in order and uses whichever is defined:

```py
for attr in ("supported_agg_types", "supported_ops", "supported_offset_types", "supported_rank_types"):
    method = getattr(self, attr, None)
    if method is not None:
        if op not in method():
            pytest.skip(f"{op} not supported by this framework")
        return
```

The multiple method names exist because different operation categories use different vocabulary: aggregation has "agg_types", binning has "ops", offset has "offset_types", rank has "rank_types". The semantics are the same.

If none of the four is defined, the helper raises `TypeError` rather than skipping. A base that declares no supported set is a defect, not a framework limitation: skipping there would make the calling shared test pass on every backend without ever running, so the coverage loss would be invisible.

---

## Example: SQLite excludes `upper`, `lower`, and `reverse`

SQLite has no native `REVERSE`, and its native `UPPER`/`LOWER` are ASCII-only and would diverge from the PyArrow reference on non-ASCII input (e.g. `UPPER('héllo')` returns `'HéLLO'` instead of `'HÉLLO'`). Rather than silently produce divergent results or emulate in SQL, the `SqliteStringOps` Implementation declares `supported` without them, so the group is not offered on SQLite for those ops (rejected at match time); another enabled framework serves them, or resolution fails:

```py
# mloda/community/feature_groups/data_operations/string/sqlite_string.py
_SQLITE_STRING_EXPRS: dict[str, str] = {
    "trim": "TRIM({col})",
    "length": "LENGTH({col})",
}


class SqliteStringOps(Implementation[StringFeatureGroup]):
    supported = frozenset(_SQLITE_STRING_EXPRS)
```

The test class declares nothing: the suite reads `SqliteStringOps.supported`, so the inherited `test_upper_*`, `test_lower_*`, and `test_reverse_*` methods skip with "not supported by this framework" instead of failing.

---

## Related

- [Framework support matrix](framework-support-matrix.md) - The capability matrix this mechanism produces, rendered as operation x framework tables.
- [Reference implementation pattern](03-reference-implementation.md) - Which reference each framework implementation is compared against.
- [String operations](09-string-operations.md) - The most common place a `supported` exclusion comes up.
- [Adding a new data operation](10-adding-new-operation.md) - How to define the hook on a new operation's test base.
