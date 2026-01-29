# Category 1: Stateless Eager Frameworks

Stateless eager frameworks execute operations immediately in-memory.

**What**: Frameworks like Pandas that evaluate operations immediately.
**When**: Standard dataframe libraries; no external connection needed.
**Why**: Simple mental model; data stays in memory.
**Where**: Pandas, PyArrow, Polars DataFrame.
**How**: Implement `is_available()`, `expected_data_framework()`, `merge_engine()`, `transform()`.

## Key Characteristic

| Property | Stateless Eager |
|----------|-----------------|
| Connection | Not required |
| Evaluation | Immediate |
| Data Location | In-memory |
| Examples | PandasDataFrame, PyArrowTable, PolarsDataFrame |

## Complete Example

```python
from typing import Any, Set, Type
from mloda.provider import ComputeFramework, BaseMergeEngine, BaseFilterEngine

try:
    import my_library as ml
except ImportError:
    ml = None


class MyEagerFramework(ComputeFramework):
    """Stateless eager framework for MyLibrary."""

    @staticmethod
    def is_available() -> bool:
        return ml is not None

    @classmethod
    def expected_data_framework(cls) -> Any:
        return ml.DataFrame  # Return TYPE, not instance

    @classmethod
    def merge_engine(cls) -> Type[BaseMergeEngine]:
        from my_plugin.my_merge_engine import MyMergeEngine
        return MyMergeEngine

    @classmethod
    def filter_engine(cls) -> Type[BaseFilterEngine]:
        from my_plugin.my_filter_engine import MyFilterEngine
        return MyFilterEngine

    def transform(self, data: Any, feature_names: Set[str]) -> Any:
        if isinstance(data, dict):
            return ml.DataFrame.from_dict(data)
        raise ValueError(f"Data type {type(data)} not supported")
```

## Test

```python
def test_my_eager_framework():
    assert MyEagerFramework.is_available()
    assert MyEagerFramework.expected_data_framework() == ml.DataFrame

    framework = MyEagerFramework(...)
    result = framework.transform({"col": [1, 2]}, {"col"})
    assert isinstance(result, ml.DataFrame)
```

## Real Implementations

| File | Description |
|------|-------------|
| [pandas/dataframe.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/pandas/dataframe.py) | Pandas DataFrame |
| [pyarrow/table.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/pyarrow/table.py) | PyArrow Table |
| [polars/dataframe.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/polars/dataframe.py) | Polars DataFrame |

## Combines With

- **Merge Engine** (Concept 6): Join operations
- **Filter Engine** (Concept 7): Filter operations
- **Transformer** (Concept 8): Cross-framework conversion
