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

```py
@classmethod
def extract_column_names(cls, data: Any) -> set[str]:
    # Get schema WITHOUT executing query
    return set(data.collect_schema().names())


def select_data_by_column_names(
    self,
    data: Any,
    selected_feature_names: Sequence[FeatureName],
    column_ordering: str | None = None,
    request_feature_order: list[str] | None = None,
) -> Any:
    column_names = set(data.collect_schema().names())
    _selected = self.identify_naming_convention(
        selected_feature_names, column_names, ordering=column_ordering, request_feature_order=request_feature_order
    )
    return data.select(list(_selected)).collect()  # Materialize HERE


@classmethod
def result_framework(cls) -> type[ComputeFramework]:
    return PolarsDataFrame  # The type select_data_by_column_names returns


def transform(self, data: Any, feature_names: Sequence[str]) -> Any:
    if isinstance(data, dict):
        return pl.LazyFrame(data)  # LazyFrame, not DataFrame
    if isinstance(data, pl.DataFrame):
        return data.lazy()  # Convert eager to lazy
    raise ValueError(f"Data type {type(data)} not supported")
```

Two result hooks sit next to `select_data_by_column_names`:

- `result_framework()` names the framework whose data type `select_data_by_column_names` returns. The default is the framework itself; override it when the selection materializes into another framework's type, as lazy Polars does (`PolarsDataFrame`). When no `output_framework` is set, plan steps and plan-lock records use it to name the result framework.
- `select_native_data_by_column_names()` is called instead of `select_data_by_column_names` when the result data is already the `output_framework`'s data type. The default delegates, so it returns whatever `select_data_by_column_names` returns (lazy Polars keeps the default and collects). Override it to return the lazy frame (`data.select(...)` without `.collect()`) when a run should get your lazy type back.

## Real Implementations

| File | Description |
|------|-------------|
| [polars/lazy_dataframe.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/polars/lazy_dataframe.py) | Polars LazyFrame |
| [polars/polars_lazy_merge_engine.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/polars/polars_lazy_merge_engine.py) | Lazy merge |

## Combines With

- **Category 1**: Inherits base structure
- **Merge Engine** (Concept 6): Lazy join operations
- **Transformer** (Concept 8): Cross-framework conversion
- **Data type extraction** (Concept 10): Implement `_extract_column_data_type()` from lazy schema metadata when
  it is available without materializing data; see [Data Type Extraction](10-data-type-extraction.md)
