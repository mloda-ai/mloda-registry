# Pattern 9: Framework-Specific Features

Framework-specific features restrict computation to certain frameworks.

**What**: Features that use framework-specific APIs (Pandas groupby, Polars expressions, Spark).
**When**: You need framework-specific optimizations or APIs not available cross-framework.
**Why**: Leverage native performance; some operations only exist in specific frameworks.
**Where**: Pandas groupby/transform, Polars lazy evaluation, DuckDB SQL, Spark distributed ops.
**How**: Return allowed frameworks from `compute_framework_rule()`.

## Key Characteristic

| Method | Behavior |
|--------|----------|
| `compute_framework_rule()` | Returns `Union[bool, Set[Type[ComputeFramework]]]` |
| Default | `True` = any framework allowed |

## Complete Example

```python
from typing import Any, Union, Set, Type
from mloda.provider import FeatureGroup, ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda.user import Feature, Options, FeatureName
from mloda.provider import FeatureSet


class PandasGroupMean(FeatureGroup):
    """Group mean using Pandas-only API."""

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PandasDataFrame}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature.not_typed("value"), Feature.not_typed("category")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.groupby("category")["value"].transform("mean")
```

## Test

```python
import pandas as pd
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

def test_pandas_group_mean():
    frameworks = PandasGroupMean.compute_framework_rule()
    assert PandasDataFrame in frameworks

    df = pd.DataFrame({"value": [1, 2, 3, 4], "category": ["A", "A", "B", "B"]})
    result = PandasGroupMean.calculate_feature(df, None)
    assert list(result) == [1.5, 1.5, 3.5, 3.5]
```

## Available Frameworks

| Framework | Import Path |
|-----------|-------------|
| Pandas | `mloda_plugins.compute_framework.base_implementations.pandas.dataframe.PandasDataFrame` |
| Polars | `mloda_plugins.compute_framework.base_implementations.polars.dataframe.PolarsDataFrame` |
| Polars Lazy | `mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe.PolarsLazyDataFrame` |
| PyArrow | `mloda_plugins.compute_framework.base_implementations.pyarrow.table.PyArrowTable` |
| DuckDB | `mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework.DuckDBFramework` |
| Spark | `mloda_plugins.compute_framework.base_implementations.spark.spark_framework.SparkFramework` |

## Real Implementations

| File | Description |
|------|-------------|
| [aggregated_feature_group/pandas.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/experimental/aggregated_feature_group/pandas.py) | Pandas aggregation |
| [aggregated_feature_group/pyarrow.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/experimental/aggregated_feature_group/pyarrow.py) | PyArrow aggregation |
| [time_window/pyarrow.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/experimental/time_window/pyarrow.py) | PyArrow time window |

## Common Pattern: Base + Framework

Define shared logic in an abstract base class, then create framework-specific subclasses:

```python
# base.py - shared pattern matching and input_features
class MyFeatureBase(FeatureGroup, ABC):
    PREFIX_PATTERN = r"^.+__my_op$"

# pandas.py - Pandas implementation
class MyFeaturePandas(MyFeatureBase):
    @classmethod
    def compute_framework_rule(cls):
        return {PandasDataFrame}

# polars.py - Polars implementation
class MyFeaturePolars(MyFeatureBase):
    @classmethod
    def compute_framework_rule(cls):
        return {PolarsDataFrame}
```

## Combines With

- **Chained** (Pattern 3): Different implementations per framework
- **Index** (Pattern 7): Framework-specific window functions
- **Artifact** (Pattern 6): Framework-specific serialization
