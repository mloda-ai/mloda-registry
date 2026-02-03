# Concept 6: Merge Engine

The merge engine handles join operations between datasets.

**What**: Component that implements JOIN and UNION operations.
**When**: Every compute framework needs a merge engine.
**Why**: Combines data from multiple feature groups during pipeline execution.
**Where**: Returned by `ComputeFramework.merge_engine()`.
**How**: Subclass `BaseMergeEngine`, implement all merge methods.

## Required Methods

| Method | SQL Equivalent | Behavior |
|--------|----------------|----------|
| `merge_inner()` | INNER JOIN | Only matching rows |
| `merge_left()` | LEFT JOIN | All left rows, matching right |
| `merge_right()` | RIGHT JOIN | All right rows, matching left |
| `merge_full_outer()` | FULL OUTER JOIN | All rows from both |
| `merge_append()` | UNION ALL | Concatenate (with duplicates) |
| `merge_union()` | UNION | Concatenate (deduplicated) |

## Complete Example

```python
from typing import Any
from mloda.provider import BaseMergeEngine
from mloda.user import Index

try:
    import pandas as pd
except ImportError:
    pd = None


class MyMergeEngine(BaseMergeEngine):
    """Merge engine for MyFramework."""

    def check_import(self) -> None:
        if pd is None:
            raise ImportError("Pandas is not installed.")

    def merge_inner(self, left_data, right_data, left_index: Index, right_index: Index) -> Any:
        return self._join("inner", left_data, right_data, left_index, right_index)

    def merge_left(self, left_data, right_data, left_index: Index, right_index: Index) -> Any:
        return self._join("left", left_data, right_data, left_index, right_index)

    def merge_right(self, left_data, right_data, left_index: Index, right_index: Index) -> Any:
        return self._join("right", left_data, right_data, left_index, right_index)

    def merge_full_outer(self, left_data, right_data, left_index: Index, right_index: Index) -> Any:
        return self._join("outer", left_data, right_data, left_index, right_index)

    def merge_append(self, left_data, right_data, left_index, right_index) -> Any:
        return pd.concat([left_data, right_data], ignore_index=True)

    def merge_union(self, left_data, right_data, left_index, right_index) -> Any:
        return self.merge_append(left_data, right_data, left_index, right_index).drop_duplicates()

    def _join(self, how: str, left_data, right_data, left_index, right_index) -> Any:
        left_on = list(left_index.index) if left_index.is_multi_index() else left_index.index[0]
        right_on = list(right_index.index) if right_index.is_multi_index() else right_index.index[0]
        return pd.merge(left_data, right_data, left_on=left_on, right_on=right_on, how=how)
```

## Test

```python
from mloda.user import Index

def test_merge_engine():
    engine = MyMergeEngine()
    left = pd.DataFrame({"idx": [1, 2], "a": ["x", "y"]})
    right = pd.DataFrame({"idx": [2, 3], "b": ["p", "q"]})
    index = Index(("idx",))

    result = engine.merge_inner(left, right, index, index)
    assert len(result) == 1  # Only idx=2 matches
```

## Stateful Connection

For frameworks with connections, use `self.framework_connection`:

```python
class MyStatefulMergeEngine(BaseMergeEngine):
    def merge_inner(self, left_data, right_data, left_index, right_index):
        conn = self.framework_connection  # Passed from framework
        # Use connection for SQL-based join...
```

## Real Implementations

| File | Description |
|------|-------------|
| [pandas/pandas_merge_engine.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/pandas/pandas_merge_engine.py) | Pandas |
| [polars/polars_merge_engine.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/polars/polars_merge_engine.py) | Polars |
| [duckdb/duckdb_merge_engine.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/duckdb/duckdb_merge_engine.py) | DuckDB |
