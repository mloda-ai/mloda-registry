# Category 2: Stateless Lazy Frameworks

Stateless lazy frameworks defer execution until results are explicitly requested.

**What**: Frameworks that build query plans before execution.
**When**: Libraries with lazy evaluation for query optimization.
**Why**: Memory-efficient processing; optimized operation ordering.
**Where**: Polars LazyFrame, Ibis, Dask.
**How**: Same as Category 1, but materialize only at end via `.collect()`.

## Key Difference from Eager

| Aspect | Eager (Category 1) | Lazy (Category 2) |
|--------|-------|------|
| Execution | Immediate | Deferred until `.collect()` |
| Schema access | `data.columns` | `data.collect_schema().names()` |
| Adding columns | Mutate in place | Return new frame (immutable) |
| Final output | Return as-is | Must `.collect()` |

## What's Different

Only these methods differ from Category 1:

```python
def set_column_names(self) -> None:
    # Get schema WITHOUT executing query
    self.column_names = set(self.data.collect_schema().names())

def select_data_by_column_names(self, data, selected_feature_names):
    column_names = set(data.collect_schema().names())
    _selected = self.identify_naming_convention(selected_feature_names, column_names)
    return data.select(list(_selected)).collect()  # Materialize HERE

def transform(self, data: Any, feature_names: Set[str]) -> Any:
    if isinstance(data, dict):
        return pl.LazyFrame(data)  # LazyFrame, not DataFrame
    if isinstance(data, pl.DataFrame):
        return data.lazy()  # Convert eager to lazy
    raise ValueError(f"Data type {type(data)} not supported")
```

## Real Implementations

| File | Description |
|------|-------------|
| [polars/lazy_dataframe.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/polars/lazy_dataframe.py) | Polars LazyFrame |
| [polars/polars_lazy_merge_engine.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/polars/polars_lazy_merge_engine.py) | Lazy merge |

## Combines With

- **Category 1**: Inherits base structure
- **Merge Engine** (Concept 6): Lazy join operations
- **Transformer** (Concept 8): Cross-framework conversion
