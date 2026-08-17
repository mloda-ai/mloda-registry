# Pattern 9: Framework-Specific Features

Framework-specific features restrict computation to certain frameworks.

**What**: Features that use framework-specific APIs (Pandas groupby, Polars expressions, Spark).
**When**: You need framework-specific optimizations or APIs not available cross-framework.
**Why**: Leverage native performance; some operations only exist in specific frameworks.
**Where**: Pandas groupby/transform, Polars lazy evaluation, DuckDB SQL, SQLite SQL, Spark distributed ops.
**How**: Return allowed frameworks from `compute_framework_rule()`.

## Key Characteristic

| Method | Behavior |
|--------|----------|
| `compute_framework_rule()` | Returns `set[type[ComputeFramework]] | None` |
| Default | `None` = any framework allowed |

## Complete Example

```python
from typing import Any
from mloda.provider import FeatureGroup, ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda.user import Feature, Options, FeatureName
from mloda.provider import FeatureSet


class PandasGroupMean(FeatureGroup):
    """Group mean using Pandas-only API."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
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
| PythonDict | `mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework.PythonDictFramework` |
| Pandas | `mloda_plugins.compute_framework.base_implementations.pandas.dataframe.PandasDataFrame` |
| Polars | `mloda_plugins.compute_framework.base_implementations.polars.dataframe.PolarsDataFrame` |
| Polars Lazy | `mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe.PolarsLazyDataFrame` |
| PyArrow | `mloda_plugins.compute_framework.base_implementations.pyarrow.table.PyArrowTable` |
| DuckDB | `mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework.DuckDBFramework` |
| SQLite | `mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework.SqliteFramework` |
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

### Column-Wise Data Hooks

A base built on `FeatureChainParserMixin` (Pattern 3) that touches columns directly, rather than returning a whole new frame, owes its framework subclasses the column-wise data hooks its `calculate_feature` calls, drawn from `_get_available_columns`, `_check_source_features_exist`, `_add_result_to_data`. The mixin defines all three as non-abstract classmethods that raise `NotImplementedError` naming the class and hook, so a framework subclass that skips one fails at compute time rather than at class-definition time.

Declare which of the three your base's `calculate_feature` actually calls with `REQUIRED_COLUMNWISE_HOOKS`, normally set to one of two constants from `mloda.provider`:

| Constant | Declares |
|----------|----------|
| `COLUMNWISE_HOOKS` | `_check_source_features_exist`, `_add_result_to_data` |
| `COLUMN_DISCOVERY_HOOKS` | those two plus `_get_available_columns`, for a base that resolves column names against the data |

```python
from mloda.provider import COLUMN_DISCOVERY_HOOKS, FeatureChainParserMixin, FeatureGroup
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


class MyFeatureBase(FeatureChainParserMixin, FeatureGroup):
    PREFIX_PATTERN = r"^.+__my_op$"
    REQUIRED_COLUMNWISE_HOOKS = COLUMN_DISCOVERY_HOOKS


class MyFeaturePandas(MyFeatureBase):
    @classmethod
    def compute_framework_rule(cls):
        return {PandasDataFrame}

    @classmethod
    def _get_available_columns(cls, data):
        return set(data.columns)

    @classmethod
    def _check_source_features_exist(cls, data, feature_names):
        if set(feature_names) - set(data.columns):
            raise ValueError(f"Missing source features, available: {list(data.columns)}")

    @classmethod
    def _add_result_to_data(cls, data, feature_name, result):
        data[feature_name] = result
        return data
```

Assert the contract holds in your own test suite instead of discovering a gap mid-run. `missing_columnwise_hooks` only checks the hooks a class declares, so pin the declaration too or a class that never sets `REQUIRED_COLUMNWISE_HOOKS` passes vacuously:

```python
from mloda.provider import COLUMN_DISCOVERY_HOOKS, missing_columnwise_hooks

def test_my_feature_pandas_implements_its_hooks():
    assert MyFeaturePandas.REQUIRED_COLUMNWISE_HOOKS == COLUMN_DISCOVERY_HOOKS
    assert missing_columnwise_hooks(MyFeaturePandas) == []
```

A hook that is not a `@classmethod` or `@staticmethod` counts as missing: `cls._hook(data, ...)` passes the class into the first parameter, so a plain function never receives the data. Assert `missing_columnwise_hooks` on the framework-bound class, not the base: the base declares the contract but implements none of it itself, so it reports every hook it declares as missing.

## Combines With

- **Chained** (Pattern 3): Different implementations per framework
- **Index** (Pattern 7): Framework-specific window functions
- **Artifact** (Pattern 6): Framework-specific serialization
