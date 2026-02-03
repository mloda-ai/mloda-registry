# Concept 8: Framework Transformer

Framework transformers enable cross-framework data conversion.

**What**: Component that converts data between frameworks (e.g., Pandas ↔ PyArrow).
**When**: You want your framework to interoperate with others.
**Why**: Enables automatic conversion chains; users can mix frameworks.
**Where**: Auto-discovered subclasses of `BaseTransformer`.
**How**: Subclass `BaseTransformer`, implement bidirectional conversion.

## PyArrow as Hub

PyArrow serves as the interchange format. Any framework can convert to any other through PyArrow:

```
Pandas ←→ PyArrow ←→ Polars
              ↕
           DuckDB
```

## Required Methods

| Method | Purpose |
|--------|---------|
| `framework()` | Returns primary framework type (e.g., `pd.DataFrame`) |
| `other_framework()` | Returns secondary type (usually `pa.Table`) |
| `import_fw()` | Import primary framework |
| `import_other_fw()` | Import secondary framework |
| `transform_fw_to_other_fw(data)` | Convert primary → secondary |
| `transform_other_fw_to_fw(data, connection)` | Convert secondary → primary |

## Complete Example

```python
from typing import Any, Optional
from mloda.provider import BaseTransformer

try:
    import my_framework as mf
except ImportError:
    mf = None

try:
    import pyarrow as pa
except ImportError:
    pa = None


class MyPyArrowTransformer(BaseTransformer):
    """Transformer: MyFramework ↔ PyArrow."""

    @classmethod
    def framework(cls) -> Any:
        return mf.DataFrame if mf else NotImplementedError

    @classmethod
    def other_framework(cls) -> Any:
        return pa.Table if pa else NotImplementedError

    @classmethod
    def import_fw(cls) -> None:
        import my_framework

    @classmethod
    def import_other_fw(cls) -> None:
        import pyarrow

    @classmethod
    def transform_fw_to_other_fw(cls, data: Any) -> Any:
        return data.to_arrow()  # MyFramework → PyArrow

    @classmethod
    def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Optional[Any] = None) -> Any:
        return mf.from_arrow(data)  # PyArrow → MyFramework
```

## Test

```python
def test_transformer():
    assert MyPyArrowTransformer.framework() == mf.DataFrame
    assert MyPyArrowTransformer.other_framework() == pa.Table

    df = mf.DataFrame({"col": [1, 2]})
    arrow = MyPyArrowTransformer.transform_fw_to_other_fw(df)
    assert isinstance(arrow, pa.Table)

    restored = MyPyArrowTransformer.transform_other_fw_to_fw(arrow)
    assert isinstance(restored, mf.DataFrame)
```

## Stateful Framework Transformers

For frameworks needing a connection (like Spark):

```python
@classmethod
def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Optional[Any] = None) -> Any:
    if framework_connection_object is None:
        raise ValueError("SparkSession required")
    return framework_connection_object.createDataFrame(data.to_pandas())
```

## Real Implementations

| File | Description |
|------|-------------|
| [pandas/pandaspyarrowtransformer.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/pandas/pandaspyarrowtransformer.py) | Pandas ↔ PyArrow |
| [polars/polars_pyarrow_transformer.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/polars/polars_pyarrow_transformer.py) | Polars ↔ PyArrow |
| [duckdb/duckdb_pyarrow_transformer.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/duckdb/duckdb_pyarrow_transformer.py) | DuckDB ↔ PyArrow |
